# Venn 1 model card

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [Adapters](adapters.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Store adapters](adapters.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · **Venn 1 model card**

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Reference / Venn 1 model card**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Intended use](#intended-use)
- [Redistribution](#redistribution)
- [Use the model](#use-the-model)
- [Training and availability](#training-and-availability)
- [Evaluation and limitations](#evaluation-and-limitations)

</details>
<!-- docs-contents:end -->

Status: research preview. The venn-1 head is published on Hugging Face at
`markrussinovich/venn-1`; pin revision
`e618a6919e5a15c6ced46c9874424363c2955283`. Venn is an embedding-native
decision layer that answers typed questions (choice, yes/no, score) over cached
embeddings and accepts the Decisions API request format of
[Jev](https://docs.typesafe.ai/introduction), TypeSafe's hosted decision model,
with its own weights and runtime.

| Property | Value |
| --- | --- |
| Profile | `venn-1` |
| Reader | `google/embeddinggemma-2` |
| Reader revision | `f9972ac22e696a105c482202646979e33e4b5a1e` |
| Vector protocol | 768-dimensional normalized text embeddings; Classification prompt |
| Generic head | `CosineAbstention`, approximately 787K parameters, 3.15 MB FP32 |
| Text-only reader | Approximately 270M parameters; about 540 MB at 16-bit, 1.08 GB in the CPU float32 path |
| Head SHA-256 | `03c62c7fbc5f5d6b49e171084c9b64291a1201b4cf967b5abbc9ebcfa3137aa9` |
| Local specialization | Residual or linear Choice families, numeric ridge/kernel families, and compiled outcome heads |
| Code license | MIT |
| Head redistribution | Permitted; see [redistribution](#redistribution) |

## Intended use

Research into typed decisions over compatible cached embeddings: conditional
choice, abstention, task-specific scoring and vector-store workflows. Validate
the actual task, label distribution and abstention policy before application use.
The reader is frozen; profile changes require new embeddings and registrations.
Google reports about 191 MB active RAM for quantized text-only weights on
Pixel 11 Pro in its [announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/).
Venn's current PyTorch reader uses the precision settings listed above;
see [deployment footprints](performance.md#model-footprint).
The outcome API can use other tagged readers; the published generic head is
tied to this profile. See [changing the reader](concepts.md#changing-the-reader).

## Redistribution

Redistribution of the published `venn-1` head weights is permitted, as confirmed
by the project owner on November 7, 2026. This permission covers the decision
head; the embedding reader and source datasets retain their own terms. The
project's source code is MIT-licensed.
The [Hub model card](https://huggingface.co/markrussinovich/venn-1#redistribution)
records the actual permission; redistribution permission alone is not a
statement of additional use or modification rights.

## Use the model

Follow the Python 3.12 environment setup in the
[installation instructions](installation.md).
From that checkout, install the reader extra and start the local Decisions API:

```bash
python -m pip install '.[embedding]'
venn download-model
venn download-embedder
venn serve
```

The head is downloaded from this Hub repository at the immutable revision
pinned in the runtime and verified against the checksums above. The frozen
EmbeddingGemma 2 reader is a separate download, not part of this head bundle.
Send the [first text decision](getting-started.md#serve-text-questions) or use the
[Python example](../examples/route_ticket.py). For a download-free CPU smoke test,
run `venn quickstart`; that example uses authored vectors, not this trained head.

## Training and availability

Behavior comes from the generic head, registered families or recorded outcomes.
The reader stays frozen. Outcome lookup changes the application-owned store;
compiling that store fits a head that can answer without a lookup.

### Generic head

The generic head was trained on CLINC, MASSIVE, Banking77, DBpedia, Yahoo
Answers and AG News. Consider this exposure when selecting held-out evaluation
tasks. The public edition includes the runtime and local family-fitting code;
the generic-head training pipeline is not public. See [licenses](licenses.md)
for source attributions and the head's redistribution permission.

Training froze the reader and used cosine option logits. The abstention MLP
was fitted by cross-entropy over K+1 choices, including none, with seed 42 for
20 epochs using AdamW (learning rate 0.001, weight decay 0.01), bf16 autocast
and gradient clipping at 1. Choice and abstention temperatures were fitted
separately on calibration episodes. TREC, GoEmotions and CLINC out-of-scope
examples were held out from head fitting and calibration.

That run reached abstention AUROC 0.929 but missed its unseen-task calibration
targets (TREC ECE 0.1215, GoEmotions ECE 0.1131), so its probabilities are not
universally calibrated.

The downloaded head is checked for integrity and profile compatibility against
the hashes above. Checksums establish artifact identity, not semantic quality.

### Local family fitting

Families fit custom predictors on your examples. The reader stays frozen.
The [family example](../examples/registered_family.py) demonstrates residual
Choice fitting; the [wine example](../examples/wine/README.md) demonstrates
numeric predictors and the [audio example](../examples/audio/README.md) fits
a linear classifier.

For residual Choice families, the generic head stays frozen.
Below 1,000 examples, option residuals use a rank-16 bilinear scorer; otherwise
they use a hidden-64 MLP over state/option/question interactions. A separate
hidden-64 network adds an abstention residual. Registered field families learn
question-conditioned field attention.

| Default | Value / boundary |
| --- | --- |
| Partitions | Every example trains by default; an optional caller `split` gives train, selection and report slices |
| Duplicate grouping | Optional `group_id` keeps related examples in one cross-validation fold and one slice |
| Minimum | Five distinct examples per label, ten overall; not certification |
| Optimizer | AdamW, learning rate 0.001, weight decay chosen from 0.001 / 0.01 / 0.1, batch 128; optional class weighting |
| Duration | Epochs (at most 24) chosen by 5-fold cross-validation; each fold stops after four non-improving epochs |
| Episodes | Variable feasible K, default 30% none when at least three labels exist |
| Calibration flag | At least 500 report items and bootstrap ECE upper bound <=0.05 |

Temperature is fitted on out-of-fold logits, or on the selection slice when the
caller gives one; a report slice is never used to select weights, settings or
temperature, and without one the family is not marked calibrated. Report uncertainty is conditional on that
family and finite examples, not evidence of generalization to new tasks.
Serving fits synchronously on CPU and persists the registration, not a modified
generic checkpoint.

A linear family (`form="linear"`) is a multinomial logistic regression on the
state vectors. It has no residual over the generic head and no abstention.

| Default | Value / boundary |
| --- | --- |
| Objective | 0.5‖W‖² + C · Σ wᵢ · cross-entropyᵢ, unpenalised intercept, float64 L-BFGS (scikit-learn's objective) |
| Penalty | C from 0.01 / 0.1 / 1 / 10 / 100 by 5-fold cross-validated log loss, grouped by `group_id` when given |
| Temperature | Fitted on out-of-fold logits, or on the selection slice when the caller gives one |
| States | Single vectors only |
| "none" | Never wins; its probability is 0 |
| Persistence | `W` and `b` in an `.npz` beside the family's JSON |

### Numeric families

`objective="regress"` fits ridge or RBF kernel ridge and returns a numeric
Score with an 80% prediction interval. The default split is 60/20/20; an explicit
split can use all supplied examples for fitting. Cross-validation selects the
penalty and, for kernel ridge, the kernel width. Optional text length inputs
and reader prompts are documented in the [API](api.md#family-lifecycle).
A kernel head persists its fitting vectors and dual weights; a linear ridge
head persists its coefficients and intercept.

### Learning from recorded outcomes

Outcome lookup performs no gradient updates; compiling a store with
`OutcomeService.register_family` does train a head. An application records a vector,
chosen option and numerical outcome, then writes them to its tagged store.
Subsequent decisions use fresh per-option neighbors. The utility definition,
feedback availability and exploration schedule are application choices, not
properties learned by Venn. `OutcomeService.register_family` can compile the
store into a bilinear head served under the same outcome question. The
[arena example](arena-example.md) demonstrates that workflow. See the [outcome learning loop](outcome-api.md#learning-without-weight-updates)
for the episodic-control convention and its limitations.

## Evaluation and limitations

The [public benchmark](benchmark.md) evaluates headless geometric operations.
It does not isolate or establish the learned head's incremental quality.
The [evidence ledger](evidence.md) lists the available results, and the
[comparison protocol](maintainers/evaluation-plan.md) states the remaining model study.

Probabilities are not universally calibrated. The fixed semantic suite exposes
number, negation and paraphrase failures. Five examples per class permit fitting,
not certification. Media is experimental. Venn is not a validated guardrail,
authorization mechanism, privacy guarantee or general reasoning engine.

<!-- docs-pager:start -->
---

[← Previous: Design rationale](design.md) · [Next: Performance and cost →](performance.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
