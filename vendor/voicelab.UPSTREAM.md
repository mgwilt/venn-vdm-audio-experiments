# VoiceLab reference submodule

Source: <https://github.com/mgwilt/voicelab>

Revision: [`00570bccf2a23f99a797ecb5db834c2f54b7b9d9`](https://github.com/mgwilt/voicelab/commit/00570bccf2a23f99a797ecb5db834c2f54b7b9d9)

Purpose: streaming audio, VAD, session lifecycle, and MLX implementation references
for the Venn audio experiments. This submodule is reference material, not a runtime
dependency or an implemented session-adaptive VAD.

## Provenance and scope

- Upstream `main` was verified with `git ls-remote` on 2026-11-10 (America/Los_Angeles).
- `.gitmodules` points `vendor/voicelab` to the upstream repository. The parent
  repository records the exact commit as a Git submodule entry, not copied source.
- The submodule is checked out at the pinned revision with no local changes.
  Its upstream licenses, notices, tests, and agent guides remain intact.
- Venn's source distribution explicitly prunes `vendor`; its Python package
  discovery includes only `venn*`.

The pinned tree has no repository-wide root `LICENSE` file. Component-specific
licenses and notices remain at their upstream paths; the host repository's MIT
license does not describe all this material. See the upstream
[model license table](voicelab/README.md#model-licenses) and each component's terms
when adapting code. No new license has been assigned to VoiceLab's own source.

## Reading map

Paths below point into the pinned submodule. Initialize it before following them.

| Concern | Source | Relevance |
| --- | --- | --- |
| Speech and turn boundaries | [turns.py](voicelab/apps/voicelab/src/voicelab/conversations/turns.py) | Silero VAD states, semantic end-of-turn analysis, half-second pre-roll, and a 30-second turn limit. Speech detection and turn completion are separate decisions. |
| Audio pipeline composition | [pipeline.py](voicelab/apps/voicelab/src/voicelab/conversations/pipeline.py) | Separates the cascade's VAD/turn aggregation from native duplex processing. |
| Worker ownership and cancellation | [processor.py](voicelab/apps/voicelab/src/voicelab/conversations/processor.py) | Serializes device work and rejects stale results using session/turn epochs. Useful for keeping future head updates from racing inference. |
| Browser microphone capture | [capture-worklet.ts](voicelab/apps/voicelab-web/frontend/audio/capture-worklet.ts), [resampler.ts](voicelab/apps/voicelab-web/frontend/audio/resampler.ts) | Captures audio, preserves fractional resampling phase across callbacks, and emits PCM16 frames at 16 or 24 kHz. Current frames are 80 ms. |
| Stateful MLX audio input | [Nemotron streaming.py](voicelab/packages/voicelab-inference/src/voicelab_inference/nemotron/mlx/streaming.py) | Explicit frame contracts, stream lifetime, PCM conversion, evaluation, and synchronization. Model-specific inference, not online head training. |
| MLX stream caches | [PersonaPlex scheduler.py](voicelab/packages/voicelab-inference/src/voicelab_inference/personaplex/mlx/scheduler.py) | Per-stream cache ownership, reset, bounded token slots, and explicit evaluation. |
| Behavioral examples | [test_pipecat_conversations.py](voicelab/tests/test_pipecat_conversations.py), [audio.test.ts](voicelab/apps/voicelab-web/tests/audio.test.ts) | Examples covering cancellation, stop hysteresis, and audio processing. Tests are preserved as references; they were not executed during vendoring. |

## Applying this to session-adaptive VAD

Use the capture, bounded buffering, state transitions, and ownership patterns as
starting points. The inspected VAD path uses existing Silero/Smart Turn analyzers;
it does not implement the proposed feedback-driven adaptive classifier.

The experiment still needs a causal feature representation, frozen baseline,
trusted feedback records, balanced session replay buffer, background linear/MLP
updates, and a safe model handoff. Evaluate a segment before training on its
feedback. Preserve the distinction between speech probability, speech boundaries,
and semantic end-of-turn decisions.

For a proposed 20–32 ms decision cadence, revisit the 80 ms capture/transport frame
contract. Subdividing a received packet does not undo the latency of collecting it.
Benchmark frame processing, onset/offset delay, missed speech, and false activations
before choosing an encoder or claiming real-time performance.

## Initialize, verify, or refresh

From the Venn repository root:

```sh
git submodule update --init --recursive vendor/voicelab
git submodule status vendor/voicelab
git -C vendor/voicelab status --short
```

To refresh, fetch upstream in the submodule, check out a deliberately selected
commit, and stage the updated `vendor/voicelab` gitlink in the parent repository.
Update the revision and reading map together. Keep experiment code outside the
submodule so local adaptations remain distinguishable from upstream source.
