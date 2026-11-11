# Audio classification

<!-- docs-navigation:start -->
[Documentation](../../docs/README.md) · [Get started](../../docs/getting-started.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md) · [Examples](../README.md) · [Task recipes](../../docs/cookbooks.md) · [Adapters](../../docs/adapters.md) · [API](../../docs/api.md) · [Results](../../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../../docs/README.md) · [Getting started](../../docs/getting-started.md) · [Installation](../../docs/installation.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../../docs/cookbooks.md) · [Store adapters](../../docs/adapters.md) · [Outcome stores](../../docs/outcome-api.md)

**Examples:** [Choose an example](../README.md) · [Hot dog image classifier](../hotdog/README.md) · [Precedent Desk](../precedent_desk/README.md) · [Wine predictor](../wine/README.md) · **Audio classifier** · [Screen-only arena](../../docs/arena-example.md) · [Demos and videos](../../docs/demos.md)

**Reference:** [API reference](../../docs/api.md) · [Design rationale](../../docs/design.md) · [Venn 1 model card](../../docs/model-card.md)

**Results and limits:** [Performance and cost](../../docs/performance.md) · [VennSpace-1](../../docs/benchmark.md) · [Results and evidence](../../docs/evidence.md) · [Reproduction](../../docs/reproduction.md) · [Semantic boundary findings](../../docs/redteam.md) · [Limitations](../../docs/limitations.md) · [Comparison video methods](../../docs/same-memory-video.md) · [Comparison notebooks and renderer](../same_memory_video/README.md)

**Project:** [Policies](../../docs/policies.md) · [Licenses and attribution](../../docs/licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../../docs/maintainers/maintaining.md) · [Documentation site](../../docs/maintainers/website.md) · [Release checklist](../../docs/maintainers/release-checklist.md) · [Publishing the Python package](../../docs/maintainers/pypi.md) · [Publishing the decision head](../../docs/maintainers/publishing-the-head.md) · [Development evidence](../../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../../docs/maintainers/evaluation-plan.md) · [License audit notes](../../docs/maintainers/license-audit.md)

</details>

**Examples / Audio classifier**
<!-- docs-navigation:end -->

## Streaming VAD on Apple Silicon

The local VAD path uses an MLX port of **Venn's linear Choice head**. It processes
32 ms frames of mono 16 kHz PCM16 audio and emits speech/background decisions,
speech boundaries, the frozen baseline score, and the active classifier version.
The audio path does not import Torch or load the text reader.

Set up on an Apple Silicon Mac with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[audio]'
venn-vad download
venn-vad devices
venn-vad mic --checkpoint .venn-audio/session.npz
```

From a source checkout, `python -m venn.audio` is equivalent to `venn-vad`.
The microphone defaults to the operating system's input device; use
`--input-device NUMBER` to choose a device from the list. macOS may request
microphone permission. Frames and events are NDJSON on stdout; redirect them to
a file to keep the terminal clear for feedback:

```bash
venn-vad mic --checkpoint .venn-audio/session.npz > .venn-audio/events.jsonl
```

While it runs, type `background 2` or `speech 2` to label the last two seconds of
processed audio. Each command labels complete frames, then attempts a bounded
training update. Use `status`, `reset` (restore the frozen baseline and clear
learning), or `quit`. Exact intervals are also accepted:

```json
{"start": 1.024, "end": 2.048, "label": "speech"}
```

The classifier **learns from explicit feedback**, not from its own predictions.
It starts with useful frozen-detector behavior, requires at least eight labeled
frames of each class before training, and retains at most 256 examples per class.
Feedback must refer to the retained last 60 seconds; relabeling the same frame
replaces its previous label. Checkpoints preserve the learned head and labeled
feature memory across sessions. They do not preserve the audio stream's recurrent
state or timeline. Features may carry information about the source audio.

For repeatable experiments, replay a WAV with optional timestamped feedback:

```bash
venn-vad file recording.wav --labels labels.json --checkpoint .venn-audio/session.npz
venn-vad benchmark --seconds 30
```

WAVs must be uncompressed mono 16 kHz PCM16. `--realtime` paces replay to the audio
clock. Without it, replay runs as fast as inference allows. A labels file is a JSON
list such as:

```json
[
  {"start": 0.0, "end": 1.024, "label": "background"},
  {"start": 2.048, "end": 3.072, "label": "speech"}
]
```

Annotations are applied only after the interval has been scored. Annotate spans
you have checked; a whole utterance may contain pauses that should remain
background. Every interval must include a full frame and fit within completed
audio; incomplete final frames are reported and discarded rather than padded.

### What is Venn, and what is the audio reader?

- The frozen Silero v6.2 ONNX reader runs with one CPU thread. Its recurrent hidden
  state, speech logit, and frame RMS produce a 130-dimensional causal feature vector.
- `MLXLinearChoiceHead` in `venn/heads/mlx_choice.py` implements the same
  `state @ W.T + b` logits and temperature-scaled softmax as Venn's existing
  `LinearChoiceHead` in `venn/heads/linear_choice.py`. Registered labels are bound
  once. The initial weights reproduce the frozen speech estimate; the Venn head
  supplies every final classification. The linear family has no learned abstention
  output: `none` remains zero and `calibrated` remains false.
- This is a new incremental SGD training path, with class balancing, clipped
  gradients, and a penalty toward the initial head. A finite candidate must lower
  replay loss before replacing active weights. This is a stability check, not
  held-out evidence of better VAD accuracy. The existing batch L-BFGS/CV fitter is
  unchanged.
- The explicit `experimental:silero-v6.2-state` profile pins the model, feature
  layout, frame length, and sample rate. These are **not `venn-1` embeddings** and
  cannot be passed to the existing generic head or registered-family HTTP path.
  The streaming API is `VADSession` in `venn/audio/session.py`.

### Latency and stream ownership

The default `--device auto` measures completed batch-one MLX inference on CPU and
Metal and picks the faster device. Force either with `--device cpu` or `--device gpu`.
This selection applies to the Venn head; the frozen ONNX reader remains on CPU.
GPU dispatch can cost more than the arithmetic for this small linear model.

The capture callback only copies audio into an eight-frame queue. A single owner
processes inference and feedback updates between frames, preventing concurrent
weight mutation. Audio overflow stops the session rather than silently dropping
samples or issuing delayed decisions on an invalid timeline. stdout backpressure
can also fill the queue; consumers should drain events promptly.

Speech onset needs two frames at probability >=0.6; speech offset needs five
frames below 0.35. Start timestamps include 160 ms of pre-roll. These are timestamp
annotations, not returned audio segments. EOF reports whether speech remains open;
it is not interpreted as natural silence. VAD does not decide semantic end-of-turn.
Frame collection, onset/offset hysteresis, capture buffers, training, and consumer
backpressure contribute latency beyond the measured inference time.

### Validation and limits

Tests cover numerical parity with the existing Torch head on CPU and Metal,
arbitrary PCM byte boundaries, hysteresis, delayed feedback, bounded memory,
corrections, learning on synthetic features, and checkpoint round-trips. Run on an
Apple Silicon process with Metal access:

```bash
python -m pytest tests/test_streaming_vad.py -q
```

An initial local test processed 456 frames from macOS-generated speech surrounded
by silence: inference p95 was 0.141 ms and maximum was 0.464 ms; one eight-step
feedback update including checkpoint save took 2.25 ms. This was a single offline
run, not a microphone latency or speech-accuracy benchmark. Synthetic-feature
tests establish that feedback changes predictions; they do not establish improved
real-world VAD. Evaluate untouched sessions and speakers before claiming that
adaptation beats the frozen baseline. `benchmark` uses synthetic silence and
reports warm inference compute only.

The pinned model is explicitly downloaded to `.venn-audio/models` with checksum
verification. Its MIT license is preserved in `venn/audio/SILERO-LICENSE`; see
[NOTICE](../../NOTICE) for source attribution.
Local models, checkpoints, and event recordings under `.venn-audio` are ignored by
Git. The pinned VoiceLab reference submodule remains unchanged; its patterns informed
stream ownership and boundary handling.

## Emotion from labeled clips

Fit a linear Choice family on your own recordings, then classify a new clip.
The reader embeds the sound; `group_id` keeps each speaker together during
cross-validation. Audio support is experimental.

From the repository root:

```bash
python -m pip install -e '.[embedding]'
venn download-embedder
python examples/audio/emotion.py --labels /path/to/clips.csv --clip /path/to/new.wav
```

The first run also downloads the Venn head. The reader runs on CPU and caches
each clip's vector. Clips must be mono 16 kHz WAV, at most 30 seconds and 6 MB.
The example does not resample or download recordings.

Create a CSV with these columns; paths are relative to the CSV:

```csv
path,label,speaker
recordings/a-angry-01.wav,angry,speaker-a
recordings/b-happy-01.wav,happy,speaker-b
```

Supply at least five distinct clips per label, at least two labels, and several
speakers represented across labels. The two rows above show the format; add your
remaining recordings before running. Keep the prediction clip out of the CSV.

The script prints JSON with the fitting clip count, cross-validation accuracy
and the new clip's `answer` and `probabilities`. It saves the family under
`VENN_HOME`, so later applications can reload it. See the
[linear-family cookbook](../../docs/cookbooks.md#fit-a-linear-choice-family)
for its request shape and the [comparison guide](../../docs/venn-and-jev.md#sound-carries-information-that-words-leave-out)
for the CREMA-D example.

<!-- docs-pager:start -->
---

[← Previous: Wine predictor](../wine/README.md) · [Next: Screen-only arena →](../../docs/arena-example.md) · [Documentation home](../../docs/README.md)
<!-- docs-pager:end -->
