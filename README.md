# Venn Audio Experiments

Streaming voice activity detection (VAD) on Apple Silicon, with a classifier that learns from feedback.
This is a fork of [Venn](https://github.com/markrussinovich/venn-vdm) focused on local audio experiments.

A frozen Silero reader extracts audio features. An MLX port of **Venn's linear Choice head** makes
speech/background decisions every 32 ms. Feedback updates the head; the audio reader stays frozen.

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
| `status` | Show the current classifier version and execution device. |
| `reset` | Clear learned corrections and restore the frozen baseline. |
| `quit` | Stop and save the checkpoint. |

Supply examples of both classes. Learning starts after at least eight labeled frames per class;
feedback can cover the last 60 seconds. The checkpoint carries the learned head and bounded example
memory into later sessions. Learning requires explicit feedback; listening alone does not train it.

## Replay and measure

Input WAVs must be uncompressed mono 16 kHz PCM16.

```sh
venn-vad file recording.wav
venn-vad file recording.wav --labels labels.json --checkpoint .venn-audio/session.npz
venn-vad benchmark --seconds 30
```

`labels.json` contains timestamped feedback, applied after each interval has been scored:

```json
[
  {"start": 0.0, "end": 1.024, "label": "background"},
  {"start": 2.048, "end": 3.072, "label": "speech"}
]
```

Use `--realtime` to pace WAV replay. `--device auto` measures MLX CPU and Metal inference and chooses
the faster path for the head; `--device cpu` or `--device gpu` overrides it. The reader runs on CPU.
The benchmark measures warm inference on synthetic silence, not microphone latency or VAD accuracy.

## Changes from Venn

| Addition | Implementation |
| --- | --- |
| MLX linear Choice head, checked against Venn's Torch implementation | `venn/heads/mlx_choice.py` |
| Stateful audio features, continuous decisions, and speech boundaries | `venn/audio/reader.py`, `venn/audio/session.py` |
| Incremental learning, bounded replay memory, and validated checkpoints | `venn/audio/session.py` |
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
python -m pytest tests/test_streaming_vad.py -q
python -m pyflakes venn scripts examples tests
python scripts/check_docs.py
python scripts/update_doc_navigation.py --check
```

The VAD tests cover CPU/Metal parity, streaming boundaries, feedback learning, checkpoints, and queue
overflow. Real microphone quality and improvement over the frozen detector still need evaluation
on held-out recordings. Local models, checkpoints, and recordings belong in the ignored `.venn-audio/` directory.

Read the [audio guide](examples/audio/README.md) for the streaming contract and validation details.
The [inherited Venn documentation](docs/README.md) covers the original APIs and examples.

## Licenses

Venn code is MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.
Silero VAD is MIT-licensed, with its license retained in `venn/audio/SILERO-LICENSE`.
VoiceLab's component licenses remain in its submodule.
