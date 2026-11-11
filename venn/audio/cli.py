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
    state = VADSession(SileroReader(model), device=args.device, checkpoint=args.checkpoint,
                       policy=getattr(args, "policy", "memory"), include_vectors=getattr(args, "vectors", False))
    if getattr(args, "questions", None):
        questions = json.loads(Path(args.questions).read_text())
        if not isinstance(questions, dict):
            raise ValueError("Questions must be a JSON object mapping names to option lists")
        for name, options in questions.items():
            state.register_question(name, options)
    return state


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


def automatic_teacher(args):
    if getattr(args, "teacher", "none") == "none":
        return None
    from venn.audio.teacher import AutoTeacher, TeacherProcess, MODEL, REVISION

    print("Loading local Parakeet teacher…", file=sys.stderr)
    teacher = AutoTeacher(TeacherProcess(args.model_dir))
    emit({"type": "teacher_ready", "model": MODEL, "revision": REVISION,
          "mode": "automatic", "labels": "conservative speech and near-silence; uncertain audio stays unlabeled"})
    return teacher


def replay(args):
    # Validate annotations before any inference or checkpoint mutation.
    with open_wav(args.wav) as source:
        duration = source.getnframes() / RATE
        labels = json.loads(Path(args.labels).read_text()) if args.labels else []
        if not isinstance(labels, list):
            raise ValueError("Labels must be a JSON list of {start, end, label} intervals")
        state = session(args)
        for row in labels:
            if (not isinstance(row, dict) or not {"start", "end", "label"} <= set(row)
                    or set(row) - {"start", "end", "label", "question"}
                    or row.get("question", "vad") not in state.questions
                    or row["label"] not in state.questions[row.get("question", "vad")].store.options
                    or not all(isinstance(row[k], (int, float)) and np.isfinite(row[k]) for k in ("start", "end"))
                    or not 0 <= row["start"] < row["end"] <= source.getnframes() // FRAME * FRAME / RATE
                    or row["end"] - row["start"] > 59):
                raise ValueError("Labels require valid completed-frame intervals of at most 59 seconds")
        labels.sort(key=lambda row: row["end"])
        emit(state.info())
        teacher = automatic_teacher(args)
        timings, frames = [], 0
        started = perf_counter()
        try:
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
                        emit(state.learn(question=row.get("question", "vad")))
                    if teacher:
                        for update in teacher.poll(state):
                            emit(update)
                        submitted = teacher.feed(pcm, state)
                        if submitted and not args.realtime:
                            # Offline replay waits for delayed supervision before advancing.
                            for update in teacher.drain(state):
                                emit(update)
            if teacher:
                for update in teacher.drain(state):
                    emit(update)
                teacher.flush(state)
                for update in teacher.drain(state):
                    emit(update)
                for update in teacher.compile(state, force=True):
                    emit(update)
        finally:
            if teacher:
                teacher.close()
            emit(state.finish())
            if state.checkpoint:
                state.save(state.checkpoint)
            emit(summary(timings, frames, perf_counter() - started), file=sys.stderr)


def command(state, line):
    """Human-friendly feedback on recent audio, or exact JSON intervals."""
    if line.startswith("{"):
        item = json.loads(line)
        if (not isinstance(item, dict) or not {"start", "end", "label"} <= set(item)
                or set(item) - {"start", "end", "label", "question"}):
            raise ValueError("Use {start, end, label} feedback with an optional question name")
        return [state.feedback(**item), state.learn(question=item.get("question", "vad"))]
    parts = line.split()
    if len(parts) == 2 and parts[0] in ("speech", "background"):
        seconds = float(parts[1])
        end = state.samples / RATE
        return [state.feedback(end - seconds, end, parts[0]), state.learn()]
    if len(parts) == 4 and parts[0] == "label":
        _, question, label, seconds = parts
        end = state.samples / RATE
        return [state.feedback(end - float(seconds), end, label, question=question), state.learn(question=question)]
    if len(parts) in (1, 2) and parts[0] == "ask":
        end = state.samples / RATE
        return [{"type": "answers", "results": state.ask(end - FRAME / RATE, end,
                 questions=parts[1:] if len(parts) == 2 else None)}]
    if parts == ["reset"]:
        state.reset_learning()
        return [{"type": "reset", "version": state.version}]
    if parts == ["status"]:
        return [state.info()]
    raise ValueError("Use: speech SECONDS | background SECONDS | label QUESTION OPTION SECONDS | ask [QUESTION] | reset | status | quit")


def microphone(args):
    import sounddevice as sd

    state = session(args)
    emit(state.info())
    teacher = automatic_teacher(args)
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

    if teacher:
        print("Automatic teaching is active. Speak normally; Ctrl-C or quit stops. Manual corrections are optional.", file=sys.stderr)
    else:
        print("Feedback: speech 2 | background 2 | label QUESTION OPTION SECONDS | ask [QUESTION] | reset | status | quit. Times refer to processed audio.", file=sys.stderr)
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
                if teacher:
                    for update in teacher.poll(state):
                        emit(update)
                try:
                    line = commands.get_nowait()
                except queue.Empty:
                    line = None
                if line == "quit":
                    # Process one already-captured frame before stopping, if available.
                    stop = True
                elif line:
                    try:
                        for event in command(state, line):
                            emit(event)
                    except (ValueError, TypeError, KeyError) as error:
                        emit({"type": "feedback_error", "message": str(error)})
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
                if teacher:
                    teacher.feed(pcm, state)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as error:
        raise RuntimeError(f"Microphone could not run: {error}") from error
    finally:
        if teacher:
            # Do not wait on transcription during shutdown. Apply completed work only.
            try:
                for update in teacher.poll(state):
                    emit(update)
                if teacher.worker.busy:
                    emit({"type": "teacher_discarded", "reason": "shutdown_with_pending_window"})
                for update in teacher.compile(state, force=True):
                    emit(update)
            finally:
                teacher.close()
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
    fetch = sub.add_parser("download", help="Download pinned reader and automatic teacher weights")
    fetch.add_argument("--model-dir", default=".venn-audio/models")
    fetch.add_argument("--teacher", choices=("parakeet", "none"), default="parakeet")
    sub.add_parser("devices", help="List microphone devices")
    for name in ("file", "mic", "benchmark"):
        child = sub.add_parser(name)
        child.add_argument("--model-dir", default=".venn-audio/models")
        child.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
        child.add_argument("--checkpoint", help="Load/save learned head and bounded labeled feature memory")
        child.add_argument("--policy", choices=("memory", "compiled"), default="memory",
                           help="Use Venn memory, or a current head compiled from its policy; unsupported VAD falls back to the frozen reader")
        child.add_argument("--questions", type=Path, help="JSON object of additional question names to option lists")
        child.add_argument("--vectors", action="store_true", help="Include reusable tagged audio vectors in frame events")
        if name != "benchmark":
            child.add_argument("--teacher", choices=("parakeet", "none"), default="parakeet",
                               help="Automatic local supervision (default); none enables manual-only operation")
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
            if args.teacher == "parakeet":
                from venn.audio.teacher import download_teacher
                emit({"teacher_model": str(download_teacher(args.model_dir))})
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
