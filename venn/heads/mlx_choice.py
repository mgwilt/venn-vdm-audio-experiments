"""MLX port of LinearChoiceHead's registered-label logits: state @ W.T + b.

The label table is bound at construction, so streaming callers do not repeat
option-vector matching. The existing Torch head remains the numerical reference.
Online SGD is an additional training path, not the batch fitter's L-BFGS/CV recipe.
"""
from time import perf_counter

import numpy as np
import mlx.core as mx


class MLXLinearChoiceHead:
    def __init__(self, W, b, labels, *, temperature=1.0, device="cpu"):
        W, b = np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32)
        if W.ndim != 2 or b.shape != (len(W),) or len(labels) != len(W):
            raise ValueError("Invalid linear Choice head dimensions")
        if len(labels) < 2 or len(set(labels)) != len(labels) or "none" in labels:
            raise ValueError("Provide distinct registered labels, excluding 'none'")
        if not np.isfinite(W).all() or not np.isfinite(b).all():
            raise ValueError("Head weights must be finite")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        if device not in ("cpu", "gpu", "auto"):
            raise ValueError("Device must be cpu, gpu, or auto")
        self.labels, self.temperature = tuple(labels), float(temperature)
        self.dimensions = W.shape[1]
        self.W, self.b = mx.array(W), mx.array(b)
        self.device_timings = {}
        if device == "auto":
            # Measure completed batch-one inference, including Python/NumPy boundaries.
            for candidate in ("cpu", "gpu"):
                if candidate == "gpu" and not mx.metal.is_available():
                    continue
                self.device = candidate
                for _ in range(5):
                    self.probabilities(np.zeros((1, self.dimensions), np.float32))
                elapsed = []
                for _ in range(30):
                    start = perf_counter()
                    self.probabilities(np.zeros((1, self.dimensions), np.float32))
                    elapsed.append((perf_counter() - start) * 1000)
                self.device_timings[candidate] = float(np.median(elapsed))
            device = min(self.device_timings, key=self.device_timings.get)
        if device == "gpu" and not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU is unavailable")
        self.device = device

    def _input(self, vectors):
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dimensions or not np.isfinite(x).all():
            raise ValueError("Expected finite [batch, dimensions] vectors")
        return x

    def logits(self, vectors):
        x = self._input(vectors)
        with mx.stream(mx.cpu if self.device == "cpu" else mx.gpu):
            result = mx.array(x) @ self.W.T + self.b
            mx.eval(result)
            return np.array(result)

    def probabilities(self, vectors):
        x = self._input(vectors)
        with mx.stream(mx.cpu if self.device == "cpu" else mx.gpu):
            result = mx.softmax((mx.array(x) @ self.W.T + self.b) / self.temperature, axis=-1)
            mx.eval(result)
            return np.array(result)

    def arrays(self):
        mx.eval(self.W, self.b)
        return np.array(self.W), np.array(self.b)

    def update(self, vectors, targets, *, anchor, steps=8, learning_rate=0.05, penalty=0.01):
        """Bounded full-replay SGD; publish only a finite candidate with lower replay loss.

        Call on the inference owner thread between frames. This check is training
        stability evidence, not a claim about unseen audio accuracy.
        """
        x = self._input(vectors)
        y = np.asarray(targets)
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("One integer label per vector is required")
        if not len(y) or y.min() < 0 or y.max() >= len(self.labels):
            raise ValueError("Invalid class index")
        if not 1 <= steps <= 32 or not 0 < learning_rate <= 1 or not 0 <= penalty <= 1:
            raise ValueError("Invalid bounded update settings")
        counts = np.bincount(y, minlength=len(self.labels))
        if (counts == 0).any():
            raise ValueError("Online training requires every class")
        with mx.stream(mx.cpu if self.device == "cpu" else mx.gpu):
            X, Y = mx.array(x), mx.array(y.astype(np.int32))
            weights = mx.array((len(y) / (len(counts) * counts[y])).astype(np.float32))
            aW, ab = (mx.array(np.asarray(a, np.float32)) for a in anchor)

            def loss(parameters):
                W, b = parameters
                scores = (X @ W.T + b) / self.temperature
                selected = mx.take_along_axis(scores, Y[:, None], axis=1).squeeze(1)
                ce = mx.mean(weights * (mx.logsumexp(scores, axis=1) - selected))
                return ce + penalty * (mx.sum((W - aW) ** 2) + mx.sum((b - ab) ** 2))

            parameters = (self.W, self.b)
            before = float(loss(parameters).item())
            gradient = mx.value_and_grad(loss)
            for _ in range(steps):
                _, grads = gradient(parameters)
                norm = mx.sqrt(sum(mx.sum(g * g) for g in grads))
                scale = mx.maximum(norm, 1.0)
                parameters = tuple(p - learning_rate * g / scale for p, g in zip(parameters, grads))
                mx.eval(*parameters)
            after = float(loss(parameters).item())
            accepted = np.isfinite(after) and after < before
            if accepted:
                self.W, self.b = parameters
        return {"accepted": bool(accepted), "loss_before": before, "loss_after": after, "steps": steps}
