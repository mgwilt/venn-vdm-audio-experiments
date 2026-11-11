"""Local Venn VAD: microphone, WAV replay, feedback, and completed-work benchmarks."""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
from time import perf_counter, sleep
import wave

import numpy as np

from venn.audio.reader import FRAME, RATE, SileroReader, download


def emit(value, *, file=sys.stdout):
    print(json.dumps(value, allow_nan=False), file=file, flush=True)


def session(args):
    from venn.audio.session import VADSession

    model = Path(args.model_dir) / "silero-v6.2.onnx"
    if not model.exists():
        raise ValueError("Missing model. Run: venn-vad download (or python -m venn.audio download)")
    return VADSession(SileroReader(model), device=args.device, checkpoint=args.checkpoint)


def open_wav(path):
    source = wave.open(str(path), "rb")
    if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (RATE, 1, 2, "NONE"):
        source.close()
        raise ValueError("WAV input must be uncompressed mono 16 kHz PCM16")
    return source


def summary(timings, frames, elapsed):
    return {"type": "summary", "frames": frames, "audio_seconds": frames * FRAME / RATE,
            "wall_seconds": elapsed, "processing_ms_p50": float(np.median(timings)) if timings else 0,
            "processing_ms_p95": float(np.percentile(timings, 95)) if timings else 0,
            "processing_ms_max": max(timings, default=0),
            "frames_over_32ms": sum(t > 32 for t in timings),
            "compute_realtime_factor": sum(timings) / 1000 / max(frames * FRAME / RATE, 1e-9)}


def replay(args):
    # Validate annotations before any inference or checkpoint mutation.
    with open_wav(args.wav) as source:
        duration = source.getnframes() / RATE
        labels = json.loads(Path(args.labels).read_text()) if args.labels else []
        if not isinstance(labels, list):
            raise ValueError("Labels must be a JSON list of {start, end, label} intervals")
        for row in labels:
            if (not isinstance(row, dict) or set(row) != {"start", "end", "label"}
                    or row["label"] not in ("speech", "background")
                    or not all(isinstance(row[k], (int, float)) and np.isfinite(row[k]) for k in ("start", "end"))
                    or not 0 <= row["start"] < row["end"] <= source.getnframes() // FRAME * FRAME / RATE
                    or row["end"] - row["start"] > 59):
                raise ValueError("Labels require valid completed-frame intervals of at most 59 seconds")
        labels.sort(key=lambda row: row["end"])
        state = session(args)
        emit(state.info())
        timings, frames = [], 0
        started = perf_counter()
        while pcm := source.readframes(FRAME):
            if args.realtime:
                delay = started + min((frames + 1) * FRAME / RATE, duration) - perf_counter()
                if delay > 0:
                    sleep(delay)
            for event in state.push(pcm):
                emit(event)
                timings.append(event["processing_ms"])
                frames += 1
                while labels and labels[0]["end"] <= event["end"] + 1e-9:
                    row = labels.pop(0)
                    emit(state.feedback(**row))
                    emit(state.learn())
        emit(state.finish())
        if state.checkpoint:
            state.save(state.checkpoint)
        emit(summary(timings, frames, perf_counter() - started), file=sys.stderr)


def command(state, line):
    """Human-friendly feedback on recent audio, or exact JSON intervals."""
    if line.startswith("{"):
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"start", "end", "label"}:
            raise ValueError("Use {start, end, label} feedback")
        return [state.feedback(**item), state.learn()]
    parts = line.split()
    if len(parts) == 2 and parts[0] in ("speech", "background"):
        seconds = float(parts[1])
        end = state.samples / RATE
        return [state.feedback(end - seconds, end, parts[0]), state.learn()]
    if parts == ["reset"]:
        state.reset_learning()
        return [{"type": "reset", "version": state.version}]
    if parts == ["status"]:
        return [state.info()]
    raise ValueError("Use: speech SECONDS | background SECONDS | reset | status | quit")


def microphone(args):
    import sounddevice as sd

    state = session(args)
    emit(state.info())
    audio = queue.Queue(maxsize=8)  # 256 ms maximum backlog; fail rather than score stale audio.
    commands = queue.Queue(maxsize=16)
    failure = threading.Event()

    def capture(data, frames, time_info, status):
        if status:
            failure.set()
            return
        try:
            audio.put_nowait(bytes(data))
        except queue.Full:
            failure.set()

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put_nowait(line.strip())
            except queue.Full:
                failure.set()
                return
            if line.strip() == "quit":
                return

    print("Feedback: speech 2 | background 2 | reset | status | quit. Times refer to processed audio.", file=sys.stderr)
    # This thread never touches the MLX model. The main worker owns all inference and updates.
    threading.Thread(target=read_commands, daemon=True).start()
    frames = 0
    timings = []
    started = perf_counter()
    try:
        with sd.RawInputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="int16",
                               device=args.input_device, callback=capture):
            stop = False
            while not stop:
                if failure.is_set():
                    raise RuntimeError("Audio overflow or callback failure; session stopped to preserve its timeline")
                try:
                    pcm = audio.get(timeout=0.1)
                except queue.Empty:
                    continue
                for event in state.push(pcm):
                    emit(event)
                    frames += 1
                    # Bound diagnostics memory independently of session length.
                    timings.append(event["processing_ms"])
                    if len(timings) > 10000:
                        del timings[:1000]
                try:
                    line = commands.get_nowait()
                except queue.Empty:
                    continue
                if line == "quit":
                    stop = True
                else:
                    try:
                        for event in command(state, line):
                            emit(event)
                    except (ValueError, TypeError, KeyError) as error:
                        emit({"type": "feedback_error", "message": str(error)})
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as error:
        raise RuntimeError(f"Microphone could not run: {error}") from error
    finally:
        emit(state.finish())
        if state.checkpoint:
            state.save(state.checkpoint)
        result = summary(timings, len(timings), perf_counter() - started)
        result.update(total_frames=frames, total_audio_seconds=frames * FRAME / RATE,
                      timing_window="most recent up to 10000 frames")
        emit(result, file=sys.stderr)


def benchmark(args):
    if not 1 <= args.seconds <= 120:
        raise ValueError("Benchmark seconds must be between 1 and 120")
    state = session(args)
    zeros = bytes(FRAME * 2)
    for _ in range(10):
        state.push(zeros)
    state.reader.reset()
    timings = []
    started = perf_counter()
    for _ in range(int(args.seconds * RATE / FRAME)):
        event = state.push(zeros)[0]
        timings.append(event["processing_ms"])
    emit({**state.info(), **summary(timings, len(timings), perf_counter() - started),
          "input": "synthetic silence", "scope": "warm inference only; excludes capture and learning"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fetch = sub.add_parser("download", help="Download pinned reader weights and license")
    fetch.add_argument("--model-dir", default=".venn-audio/models")
    sub.add_parser("devices", help="List microphone devices")
    for name in ("file", "mic", "benchmark"):
        child = sub.add_parser(name)
        child.add_argument("--model-dir", default=".venn-audio/models")
        child.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
        child.add_argument("--checkpoint", help="Load/save learned head and bounded labeled feature memory")
        if name == "file":
            child.add_argument("wav", type=Path)
            child.add_argument("--labels", type=Path, help="JSON intervals; applied only after their audio was scored")
            child.add_argument("--realtime", action="store_true")
        elif name == "mic":
            child.add_argument("--input-device", type=int)
        else:
            child.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        if args.mode == "download":
            emit({"model": str(download(args.model_dir))})
        elif args.mode == "devices":
            import sounddevice as sd
            print(sd.query_devices())
        elif args.mode == "file":
            replay(args)
        elif args.mode == "mic":
            microphone(args)
        else:
            benchmark(args)
    except (ValueError, RuntimeError, OSError, wave.Error) as error:
        parser.exit(1, f"venn-vad: {error}\n")


if __name__ == "__main__":
    main()
