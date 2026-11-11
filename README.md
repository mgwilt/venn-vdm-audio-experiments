# Venn Audio Experiments

Venn for streaming audio on Apple Silicon. This fork of [Venn](https://github.com/markrussinovich/venn-vdm)
tests whether reusable audio vectors and editable session memory can support decisions that adapt during
use. Voice activity detection (VAD) is the first task.

Venn separates the audio representation from the decisions made over it. The same vectors can answer
multiple questions, corrections become retrievable evidence, and small MLX heads learn from memory's
decisions. That makes session adaptation possible without retraining the reader for each question.

## Decision loop

```text
audio → tagged vectors → Venn memory → decisions + evidence
  └→ background teacher → labeled cases ↗  └→ compiled MLX heads
```

Venn's reader encodes each 32 ms frame once. Its `OutcomeService` retrieves nearby labeled cases for each
registered question. New labels change those answers immediately; head compilation learns the resulting
memory policy. Questions can be re-evaluated over retained vectors without processing the audio again.

Default `--policy memory` queries cases on every frame. `--policy compiled` uses a current head and its
saved coverage check without querying the live store. New feedback invalidates stale heads until
recompilation succeeds. Unsupported questions abstain; VAD has a frozen-detector fallback.

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

`download` fetches and checksum-verifies the reader and Parakeet teacher (about 2.3 GB of teacher weights).
Choose a microphone with
`--input-device NUMBER` if the system default is unsuitable. Events are written as newline-delimited
JSON; inspect them in another terminal with `tail -f .venn-audio/events.jsonl`.

## Automatic teaching

Speak normally. Parakeet runs in a separate process over overlapping audio windows, including audio
Venn currently rejects. Accepted speech spans and sustained near-silence become labeled memory cases;
heads compile periodically. No manual labeling is required. Stop with Ctrl-C or `quit`.

Uncertain audio stays unlabeled. Empty transcripts do not imply background, and ordinary room noise
is not automatically labeled as silence. Each case retains its label source. Manual corrections take
priority; `reset` clears memory and invalidates pending teacher results. Use `--teacher none` to disable
automatic teaching on `mic` or `file` (also available on `download` to fetch only the reader).

Optional commands while the microphone runs:

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
to 256 examples per label. Checkpoints preserve questions, cases, sources, and heads across sessions.
Automatic teaching currently labels the VAD question; other questions need their own examples.

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
venn-vad file recording.wav --teacher none --policy compiled --checkpoint .venn-audio/session.npz
venn-vad benchmark --seconds 30
```

`labels.json` contains timestamped feedback, applied after each interval has been scored:

```json
[
  {"start": 0.0, "end": 1.024, "label": "background"},
  {"start": 2.048, "end": 3.072, "label": "speech"}
]
```

Labels may include a `question` field to target a registered question. Offline replay waits for teacher
results between windows; `--realtime` paces audio and skips teacher windows if the worker falls behind.
Live decisions never wait for transcription. Use `--teacher none` for a fixed-checkpoint evaluation.
`--device auto` measures MLX CPU and Metal inference and chooses
the faster path for the head; `--device cpu` or `--device gpu` overrides it. The reader runs on CPU.
The benchmark measures warm inference on synthetic silence, not microphone latency or VAD accuracy.

## Changes from Venn

| Addition | Implementation |
| --- | --- |
| Venn outcome-store adapter, evidence retrieval, and memory-policy targets | `venn/audio/memory.py` |
| Automatic Parakeet supervision, conservative labels, and bounded worker queues | `venn/audio/teacher.py` |
| MLX linear Choice head, checked against Venn's Torch implementation | `venn/heads/mlx_choice.py` |
| Tagged audio vectors, continuous decisions, and speech boundaries | `venn/audio/reader.py`, `venn/audio/session.py` |
| Named questions, head compilation, and versioned memory checkpoints | `venn/audio/session.py` |
| Microphone capture, WAV replay, feedback, and benchmarks | `venn/audio/cli.py` / `venn-vad` |
| Pinned VoiceLab reference submodule | `vendor/voicelab` |

The current reader uses frozen Silero features under an experimental audio vector profile. These are
not `venn-1` embeddings. The streaming runtime is exposed through `venn-vad` and `VADSession`, separately
from Venn's existing HTTP family API. The inherited `venn` CLI and batch classifiers remain available;
the audio process does not load Torch or the text reader.

[VoiceLab](https://github.com/mgwilt/voicelab) supplies implementation references, not a runtime dependency.
For an existing clone, initialize it with `git submodule update --init --recursive`.

## Develop

With the environment above activated:

```sh
python -m pytest tests/test_audio_memory.py tests/test_audio_teacher.py tests/test_streaming_vad.py -q
python -m pyflakes venn scripts examples tests
python scripts/check_docs.py
python scripts/update_doc_navigation.py --check
```

Tests cover CPU/Metal parity, immediate corrections, shared-vector questions, evidence abstention,
compiled inference without live-store queries, automatic labels, manual precedence, stale teacher results,
checkpoint migration, streaming boundaries, and queue overflow.
Improvement over fixed VAD still needs evaluation on held-out recordings.
If macOS reports duplicate OpenMP runtimes when testing Torch/FAISS together, run
`DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib" python -m pytest -q`.
Local models, checkpoints, and recordings belong in the ignored `.venn-audio/` directory.

Read the [audio guide](examples/audio/README.md) for the streaming contract and validation details.
The [inherited Venn documentation](docs/README.md) covers the original APIs and examples.

## Licenses

Venn code is MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.
Silero VAD is MIT-licensed, with its license retained in `venn/audio/SILERO-LICENSE`.
Parakeet weights are CC BY 4.0; the `parakeet-mlx` runtime is Apache-2.0. See [NOTICE](NOTICE).
VoiceLab's component licenses remain in its submodule.
