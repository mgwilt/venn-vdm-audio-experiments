# Venn Audio Experiments

An experiment in adapting voice activity detection (VAD) to a session without retraining the audio reader.
This fork of [Venn](https://github.com/markrussinovich/venn-vdm) uses its core pattern: encode audio once,
reuse the vectors across questions, make decisions from remembered examples, and compile those decisions
into small MLX heads for Apple Silicon.

A frozen Silero reader produces a tagged vector every 32 ms. Venn's `OutcomeService` queries labeled
session memory and returns supporting case IDs. Corrections affect memory immediately; unsupported or
conflicting VAD evidence falls back to Silero. Whether this improves accuracy is still an open experiment.

## Start

Requires an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/). Run from a terminal with microphone access.

```sh
git clone --recurse-submodules https://github.com/mgwilt/venn-vdm-audio-experiments.git
cd venn-vdm-audio-experiments
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[audio,dev]'
venn-vad download
venn-vad devices
venn-vad mic --checkpoint .venn-audio/session.npz > .venn-audio/events.jsonl
```

`download` fetches the pinned Silero model and verifies its checksum. Choose a microphone with
`--input-device NUMBER` if the system default is unsuitable. Events are written as newline-delimited
JSON; inspect them in another terminal with `tail -f .venn-audio/events.jsonl`.

## Teach it

While the microphone runs, type feedback into the same terminal:

| Command | Effect |
| --- | --- |
| `speech 2` | Label the preceding two seconds of processed audio as speech. |
| `background 2` | Label the preceding two seconds as background. |
| `ask` | Re-query the latest vector with current memory or heads. |
| `status` | Show question counts, memory/head revisions, and execution device. |
| `reset` | Clear learned corrections and restore the frozen baseline. |
| `quit` | Stop and save the checkpoint. |

Feedback can cover the last 60 seconds. Each command stores corrections, then tries to compile memory's
decisions into a head; compilation needs at least eight supported targets per class. Memory retains up
to 256 examples per label. Checkpoints preserve questions, cases, and heads across sessions. Listening
alone does not train anything.

Default `--policy memory` queries examples on every frame. Use `--policy compiled` to run a current MLX
head with its saved coverage check, without querying the live store. New feedback makes that head stale
until recompilation succeeds; inference uses memory in the meantime.

## Reuse the vectors

Define extra questions in `questions.json`, for example `{"sound": ["fan", "keyboard"]}`:

```sh
venn-vad mic --questions questions.json --vectors --checkpoint .venn-audio/session.npz > .venn-audio/events.jsonl
```

Type `label sound keyboard 2` or `label sound fan 2` to teach that question, then `ask sound` to query it.
All questions share the reader's vectors; additional questions abstain until supported. Their accuracy
depends on what the frozen speech features can distinguish.

Protocol v2 events include `state`, named `questions`, and a `decision` alias for VAD. `--vectors` adds
the reusable vector to each state. Memory answers expose agreement `scores`, evidence, and
`probabilities: null`; model probabilities are uncalibrated. Python callers can use
`VADSession.register_question()`, `feedback()`, `ask()`, and `learn()` without re-encoding retained audio.

## Replay and measure

Input WAVs must be uncompressed mono 16 kHz PCM16.

```sh
venn-vad file recording.wav
venn-vad file recording.wav --labels labels.json --checkpoint .venn-audio/session.npz
venn-vad file recording.wav --policy compiled --checkpoint .venn-audio/session.npz
venn-vad benchmark --seconds 30
```

`labels.json` contains timestamped feedback, applied after each interval has been scored:

```json
[
  {"start": 0.0, "end": 1.024, "label": "background"},
  {"start": 2.048, "end": 3.072, "label": "speech"}
]
```

Labels may include a `question` field to target a registered question. Use `--realtime` to pace WAV replay.
`--device auto` measures MLX CPU and Metal inference and chooses
the faster path for the head; `--device cpu` or `--device gpu` overrides it. The reader runs on CPU.
The benchmark measures warm inference on synthetic silence, not microphone latency or VAD accuracy.

## Changes from Venn

| Addition | Implementation |
| --- | --- |
| Venn outcome-store adapter, evidence retrieval, and memory-policy targets | `venn/audio/memory.py` |
| MLX linear Choice head, checked against Venn's Torch implementation | `venn/heads/mlx_choice.py` |
| Tagged audio vectors, continuous decisions, and speech boundaries | `venn/audio/reader.py`, `venn/audio/session.py` |
| Named questions, head compilation, and versioned memory checkpoints | `venn/audio/session.py` |
| Microphone capture, WAV replay, feedback, and benchmarks | `venn/audio/cli.py` / `venn-vad` |
| Pinned VoiceLab reference submodule | `vendor/voicelab` |

The streaming path uses its own audio feature profile, not `venn-1` embeddings. It runs locally through
`venn-vad` or `VADSession`; it is not wired into Venn's existing HTTP family API. The inherited `venn`
CLI and batch classifiers remain available. The audio process does not load Torch or the text reader.

[VoiceLab](https://github.com/mgwilt/voicelab) supplies implementation references, not a runtime dependency.
For an existing clone, initialize it with `git submodule update --init --recursive`.

## Develop

With the environment above activated:

```sh
python -m pytest tests/test_audio_memory.py tests/test_streaming_vad.py -q
python -m pyflakes venn scripts examples tests
python scripts/check_docs.py
python scripts/update_doc_navigation.py --check
```

Tests cover CPU/Metal parity, immediate corrections, shared-vector questions, evidence abstention,
compiled inference without live-store queries, checkpoint migration, streaming boundaries, and queue overflow.
If macOS reports duplicate OpenMP runtimes when testing Torch/FAISS together, run
`DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib" python -m pytest -q`.
Local models, checkpoints, and recordings belong in the ignored `.venn-audio/` directory.

Read the [audio guide](examples/audio/README.md) for the streaming contract and validation details.
The [inherited Venn documentation](docs/README.md) covers the original APIs and examples.

## Licenses

Venn code is MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.
Silero VAD is MIT-licensed, with its license retained in `venn/audio/SILERO-LICENSE`.
VoiceLab's component licenses remain in its submodule.
