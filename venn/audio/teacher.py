"""Delayed Parakeet supervision; live decisions never wait for transcription.

Raw PCM lives only in bounded windows/IPC. The spawned process owns Parakeet;
the session owner alone applies its results. Labels are conservative pseudo-labels,
not ground truth. Missing transcripts never imply background.
"""
import hashlib
import multiprocessing as mp
from pathlib import Path
import queue
from time import monotonic, perf_counter, sleep

import numpy as np

from venn.audio.reader import FRAME, RATE

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
MODEL_FOLDER = "parakeet-tdt-0.6b-v3"
HASHES = {
    "config.json": "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c",
    "model.safetensors": "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592",
    "README.md": "724e3ba46903986ff6caeb8fddb7495c06b5a7a8d908f11e43cc0a7e136b5e23",
}
SOURCE = f"parakeet:{REVISION}"
SILENCE_SOURCE = "acoustic:near-silence-v1"
WINDOW_FRAMES = 128  # 4.096 seconds, independent of the live VAD gate.
HOP_FRAMES = 64
MIN_CONFIDENCE = .9  # Parakeet entropy score, NOT a calibrated probability.
SILENCE_RMS = 1e-4  # -80 dBFS, intentionally excludes ordinary room noise.
SILENCE_PEAK = 10 ** (-70 / 20)
GUARD_FRAMES = 5  # 160 ms around tokens and window edges.


def verify_model(directory):
    path = Path(directory) / MODEL_FOLDER
    for name, expected in HASHES.items():
        target = path / name
        if not target.is_file():
            raise ValueError("Missing Parakeet teacher. Run: venn-vad download")
        with target.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Teacher checksum mismatch: {target}")
    return path


def download_teacher(directory):
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION, local_dir=Path(directory) / MODEL_FOLDER,
                      allow_patterns=list(HASHES))
    return verify_model(directory)


def label_window(pcm, tokens):
    """Select complete frames inside confident tokens or sustained near-silence.

    Trim token boundaries by a frame; exclude window edges and clipping. Quiet
    negatives need eight contiguous frames and distance from *all* ASR tokens,
    including uncertain tokens. No current VAD score is consulted.
    """
    if len(pcm) % (FRAME * 2) or not FRAME * 2 <= len(pcm) <= WINDOW_FRAMES * FRAME * 2:
        raise ValueError("Teacher windows require 1..128 complete PCM16 frames")
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(-1, FRAME) / 32768
    rms = np.sqrt(np.mean(audio * audio, axis=1))
    peak = np.max(np.abs(audio), axis=1)
    count, duration = len(audio), len(audio) * FRAME / RATE
    starts = np.arange(count) * FRAME / RATE
    ends = starts + FRAME / RATE
    speech = np.zeros(count, bool)
    near_token = np.zeros(count, bool)
    confidence = np.zeros(count)
    for token in tokens:
        start, end, score = (token[k] for k in ("start", "end", "confidence"))
        if (not all(np.isfinite(v) for v in (start, end, score)) or not 0 <= score <= 1
                or not 0 <= start <= end <= duration + .16 or not isinstance(token["text"], str)):
            raise ValueError("Invalid Parakeet token timing or confidence")
        near_token |= (ends > start - GUARD_FRAMES * FRAME / RATE) & (starts < end + GUARD_FRAMES * FRAME / RATE)
        if score < MIN_CONFIDENCE or not any(c.isalnum() for c in token["text"]) or not 0 < end - start <= 1:
            continue
        interior = (starts >= start + FRAME / RATE) & (ends <= end - FRAME / RATE)
        speech |= interior
        confidence[interior] = np.maximum(confidence[interior], score)
    speech &= (peak > SILENCE_PEAK) & (peak < .999)
    quiet = (rms <= SILENCE_RMS) & (peak <= SILENCE_PEAK) & ~near_token
    speech[:GUARD_FRAMES] = speech[-GUARD_FRAMES:] = False
    quiet[:GUARD_FRAMES] = quiet[-GUARD_FRAMES:] = False
    spans = []
    for label, mask, minimum, source in (("speech", speech, 1, SOURCE), ("background", quiet, 8, SILENCE_SOURCE)):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for begin, finish in zip(edges[::2], edges[1::2]):
            if finish - begin >= minimum:
                spans.append({"start": int(begin) * FRAME / RATE, "end": int(finish) * FRAME / RATE,
                              "label": label, "source": source,
                              "teacher_score": float(confidence[begin:finish].min()) if label == "speech" else None})
    return sorted(spans, key=lambda span: span["start"])


class ParakeetTeacher:
    def __init__(self, directory):
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(str(verify_model(directory).resolve()))
        if self.model.preprocessor_config.sample_rate != RATE:
            raise ValueError("Teacher must accept 16 kHz audio")

    def transcribe(self, pcm):
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        result = self.model.generate(get_logmel(mx.array(audio), self.model.preprocessor_config))[0]
        return {"text": result.text, "tokens": [{"start": float(t.start), "end": float(t.end),
                "text": t.text, "confidence": float(t.confidence)} for t in result.tokens]}


def _worker(directory, jobs, results):
    try:
        teacher = ParakeetTeacher(directory)
        teacher.transcribe(bytes(WINDOW_FRAMES * FRAME * 2))  # Complete lazy model/Metal initialization.
        results.put({"type": "teacher_ready", "model": MODEL, "revision": REVISION})
        while (job := jobs.get()) is not None:
            started = perf_counter()
            output = teacher.transcribe(job["pcm"])
            spans = label_window(job["pcm"], output["tokens"])
            results.put({"type": "teacher_result", **{k: v for k, v in job.items() if k != "pcm"},
                         "text": output["text"], "spans": spans, "model": MODEL, "revision": REVISION,
                         "processing_ms": (perf_counter() - started) * 1000})
    except Exception as error:
        results.put({"type": "teacher_error", "message": f"{type(error).__name__}: {error}"})


class TeacherProcess:
    """One in-flight window, bounded IPC, independent MLX ownership."""
    def __init__(self, directory):
        context = mp.get_context("spawn")
        self.jobs, self.results = context.Queue(maxsize=1), context.Queue(maxsize=1)
        self.process = context.Process(target=_worker, args=(str(directory), self.jobs, self.results), daemon=True)
        self.busy = False
        self.failed = False
        self.process.start()
        try:
            ready = self.results.get(timeout=120)
            if ready["type"] != "teacher_ready":
                raise RuntimeError(ready.get("message", "Teacher failed to start"))
        except queue.Empty:
            self.close()
            raise RuntimeError("Parakeet startup timed out; check Metal access") from None
        except BaseException:
            self.close()
            raise

    def submit(self, job):
        if self.busy or self.failed:
            return False
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            return False
        self.busy = True
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive() and not self.failed:
                self.failed, self.busy = True, False
                return {"type": "teacher_error", "message": "Parakeet worker exited; automatic teaching stopped"}
            return None
        self.busy = False
        if result["type"] == "teacher_error":
            self.failed = True
        return result

    def close(self):
        if self.process.is_alive():
            try:
                self.jobs.put_nowait(None)
            except queue.Full:
                pass
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        for channel in (self.jobs, self.results):
            channel.cancel_join_thread()
            channel.close()


class AutoTeacher:
    """Session-owner controller; worker results only modify retained, past frames."""
    def __init__(self, worker):
        self.worker = worker
        self.buffer = bytearray()
        self.identity = None
        self.start_sample = 0
        self.last_submitted = -1
        self.next_compile = 0
        self.last_attempt_revision = -1
        self.skipped_windows = 0

    def feed(self, pcm, state):
        if not isinstance(pcm, bytes) or len(pcm) % (FRAME * 2):
            raise ValueError("Feed completed PCM16 frames to the teacher")
        identity = (state.session_id, state.learning_epoch)
        start = state.samples - len(pcm) // 2
        if identity != self.identity or start != self.start_sample + len(self.buffer) // 2:
            self.buffer.clear()
            self.start_sample, self.identity = start, identity
            self.next_compile, self.last_attempt_revision = 0, -1
        self.buffer.extend(pcm)
        submitted = False
        size = WINDOW_FRAMES * FRAME * 2
        while len(self.buffer) >= size:
            submitted |= self._submit(bytes(self.buffer[:size]), state)
            del self.buffer[:HOP_FRAMES * FRAME * 2]
            self.start_sample += HOP_FRAMES * FRAME
        return submitted

    def _submit(self, pcm, state):
        end = self.start_sample + len(pcm) // 2
        if end <= self.last_submitted:
            return False
        job = {"session_id": state.session_id, "epoch": state.learning_epoch,
               "start_sample": self.start_sample, "end_sample": end, "pcm": pcm}
        if not self.worker.submit(job):
            self.skipped_windows += 1
            return False
        self.last_submitted = end
        return True

    def flush(self, state):
        # Do not pad EOF or reuse audio preceding reset.
        if self.identity == (state.session_id, state.learning_epoch) and len(self.buffer) >= 16 * FRAME * 2:
            return self._submit(bytes(self.buffer), state)
        return False

    def poll(self, state):
        result = self.worker.poll()
        if result is None:
            return []
        if result["type"] == "teacher_error":
            return [result]
        result["skipped_windows"] = self.skipped_windows
        if (result["session_id"], result["epoch"]) != (state.session_id, state.learning_epoch):
            return [{"type": "teacher_discarded", "reason": "session_or_reset_changed"}]
        offset = result["start_sample"] / RATE
        group = f"teacher:{state.session_id}:{result['epoch']}:{result['end_sample']}"
        events, changed = [], 0
        for span in result["spans"]:
            try:
                event = state.feedback(offset + span["start"], offset + span["end"], span["label"],
                                       source=span["source"], group=group)
            except ValueError:  # A slow result can outlive retained history. Never shift its timestamps.
                continue
            changed += event["changed_frames"]
            events.append(event)
        result["changed_frames"] = changed
        events.insert(0, result)
        events.extend(self.compile(state))
        return events

    def compile(self, state, *, force=False):
        revision = state.questions["vad"].store.revision
        if revision != self.last_attempt_revision and (force or state.samples >= self.next_compile):
            event = state.learn()
            self.last_attempt_revision = revision
            self.next_compile = state.samples + 5 * RATE
            return [event]
        return []

    def drain(self, state, timeout=30):
        deadline = monotonic() + timeout
        events = []
        while self.worker.busy and monotonic() < deadline:
            events.extend(self.poll(state))
            if self.worker.busy:
                sleep(.01)
        if self.worker.busy:
            raise RuntimeError("Parakeet teacher timed out; no labels inferred from missing results")
        return events

    def close(self):
        self.worker.close()
        self.buffer.clear()
