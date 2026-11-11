"""Pinned, frozen Silero ONNX reader; direct NumPy I/O without importing Torch.

Silero v6.2 model interface: 512 new samples, 64 context samples, recurrent
state [2, 1, 128]. See the pinned upstream utils_vad.py and MIT license.
"""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np

from venn.profiles import VectorTag

REVISION = "be95df9152c0d7618fa1edfeb296fc3dae32376f"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
LICENSE_SHA256 = "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b"
RATE = 16000
FRAME = 512
DIMENSIONS = 130
PROFILE = {
    "profile": "experimental:silero-v6.2-state",
    "revision": REVISION,
    "model_sha256": MODEL_SHA256,
    "representation": "hidden128-logit-div8-rms-v1",
    "dimensions": DIMENSIONS,
    "sample_rate": RATE,
    "frame_samples": FRAME,
}
AUDIO_TAG = VectorTag(PROFILE["profile"], REVISION, PROFILE["representation"], DIMENSIONS, ("audio",))


def _verify(path, digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch: {path}")


def download(directory):
    """Explicit download command; verify model and license before atomic replacement."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/snakers4/silero-vad/{REVISION}/"
    for remote, name, digest in (
        ("src/silero_vad/data/silero_vad.onnx", "silero-v6.2.onnx", MODEL_SHA256),
        ("LICENSE", "SILERO-LICENSE", LICENSE_SHA256),
    ):
        destination = directory / name
        if destination.exists():
            _verify(destination, digest)
            continue
        temporary = None
        try:
            with urlopen(base + remote, timeout=60) as response, tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                data = response.read(4_000_001)
                if len(data) > 4_000_000:
                    raise ValueError("Unexpected model download size")
                output.write(data)
            _verify(temporary, digest)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory / "silero-v6.2.onnx"


class SileroReader:
    def __init__(self, path):
        import onnxruntime as ort

        _verify(path, MODEL_SHA256)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def encode(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        if x.shape != (FRAME,) or not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
            raise ValueError("Reader requires 512 finite mono float samples in [-1, 1]")
        audio = np.concatenate((self.context, x[None]), axis=1)
        probability, state = self.session.run(None, {
            "input": audio, "state": self.state, "sr": np.array(RATE, np.int64),
        })
        p = float(probability.item())
        if not np.isfinite(p) or not 0 <= p <= 1 or not np.isfinite(state).all():
            raise ValueError("Reader produced invalid features")
        self.state = state
        self.context = audio[:, -64:].copy()
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)) / 8
        rms = np.sqrt(np.mean(x * x))
        energy = np.clip((20 * np.log10(max(float(rms), 1e-6)) + 60) / 60, -1, 1)
        features = np.concatenate((state[0, 0], [logit, energy])).astype(np.float32)
        return features, p
