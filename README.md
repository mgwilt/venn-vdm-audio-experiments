# Venn Audio Experiments

This fork of [Venn](https://github.com/markrussinovich/venn-vdm) adds streaming audio classification on Apple Silicon.
Venn reuses audio vectors across questions, stores labeled examples, and trains small MLX heads from memory decisions.
Voice activity detection (VAD) is the first task.

## Start

Use an Apple Silicon Mac with [uv](https://docs.astral.sh/uv/).
Give the terminal microphone access.

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

The download includes the reader and about 2.3 GB of Parakeet weights.
Checksums verify both models.
Use `--input-device NUMBER` to select a microphone.

Speak normally. Parakeet labels speech and sustained near-silence in the background.
Venn adds accepted labels to memory and trains heads at intervals.
Manual labels are optional. Press Ctrl-C or type `quit` to stop and save.

To inspect events, run this command in another terminal:

```sh
tail -f .venn-audio/events.jsonl
```

## Run

Use uncompressed mono 16 kHz PCM16 WAV files.

```sh
venn-vad file recording.wav --checkpoint .venn-audio/session.npz
venn-vad file recording.wav --teacher none --policy compiled --checkpoint .venn-audio/session.npz
venn-vad benchmark --seconds 30
```

| Option | Function |
| --- | --- |
| `--teacher none` | Disable automatic labels. |
| `--policy memory` | Query stored examples. This is the default. |
| `--policy compiled` | Use a current MLX head. Use memory if the head is out of date. |
| `--questions questions.json` | Add questions that use the same audio vectors. |
| `--vectors` | Include vectors in JSON events. |
| `--realtime` | Process a WAV at its recorded rate. |
| `--device auto` | Select the faster MLX device. Use `cpu` or `gpu` to override. |

Offline replay waits for teacher results. Live decisions do not wait for transcription.
The benchmark measures inference on synthetic silence. It does not measure VAD accuracy.

## Memory and labels

Each 32 ms frame produces a tagged vector.
Venn's `OutcomeService` returns decisions with supporting case IDs.
New labels change memory answers immediately. Heads learn those answers.

Uncertain audio stays unlabeled. An empty transcript does not prove that speech is absent.
Automatic labels apply only to VAD. Other questions need separate examples.
Human corrections take priority over automatic labels.

During capture, type `speech 2` or `background 2` to correct the last two seconds of processed audio.
Use `status` to inspect the session. Use `reset` to clear learned data.
Feedback can cover the last 60 seconds. Memory stores up to 256 examples per label.

Questions without enough evidence return `none`. VAD uses the frozen detector as a fallback.
Memory scores are label agreement, not probabilities. Model probabilities are not calibrated.
Improved accuracy still needs testing on held-out recordings.

See the [audio guide](examples/audio/README.md) for question definitions, label files, Python APIs, and event fields.

## Changes from Venn

| Component | Source |
| --- | --- |
| Audio memory and evidence queries | `venn/audio/memory.py` |
| Automatic Parakeet labels | `venn/audio/teacher.py` |
| MLX linear Choice head | `venn/heads/mlx_choice.py` |
| Audio vectors, questions, boundaries, and checkpoints | `venn/audio/reader.py`, `venn/audio/session.py` |
| Microphone and file commands | `venn/audio/cli.py` |

The current reader uses frozen Silero features. These vectors are not compatible with `venn-1`.
The streaming API is separate from Venn's HTTP API.
[VoiceLab](https://github.com/mgwilt/voicelab) is a reference submodule at `vendor/voicelab`. It is not a runtime dependency.

## Develop

```sh
python -m pytest tests/test_audio_memory.py tests/test_audio_teacher.py tests/test_streaming_vad.py -q
python -m pyflakes venn scripts examples tests
python scripts/check_docs.py
python scripts/update_doc_navigation.py --check
```

Store local models, checkpoints, and recordings in `.venn-audio/`. Git ignores this directory.
See the [Venn documentation](docs/README.md) for inherited APIs.

## Licenses

Venn and Silero code: MIT. Parakeet weights: CC BY 4.0. `parakeet-mlx`: Apache-2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE). VoiceLab retains its component licenses in the submodule.
