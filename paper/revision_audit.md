# IEEE Access revision audit: Access-2026-42627

**Status: working revision, not submission-ready.** No new five-seed benchmark
results or comparative GPU profiles have been generated. This is an editorial
and technical assessment, not an IEEE decision or acceptance prediction.

## Editorial judgment

[Likely] Revision strategy: **9/10** if the missing evidence is completed and
reported honestly. Current scientific submission readiness: **3/10**. Improving
the wording cannot replace the spatial, repeated-run, profiling, and controlled
ablation results requested by Reviewer 2. Five seeds address variability; they
are neither an IEEE-wide fixed requirement nor a guarantee of acceptance.

[Certain] Reviewer 1 permits narrower claims and allows some new experiments to
remain future work. Reviewer 2 separately asks for computational measurements
and multiple independent runs, and strongly recommends a spatial protocol.
An answer to Reviewer 1 does not discharge Reviewer 2's request.

## How the scores work

Priority measures the risk of leaving a concern unresolved (10 is highest).
Response and manuscript scores measure present directness, technical accuracy,
evidence, traceability and limitation disclosure, each worth two points.
They are **provisional author-side editorial judgments**, not future-result
scores. Experimental concerns remain low even when their draft response is clear.

| ID | Category / disposition | Priority | Original response / manuscript | Draft response / manuscript | Remaining requirement |
|---|---|---:|---:|---:|---|
| R1-C1 | Protocol transparency; must correct | 10 | 0 / 2 | 7 / 7 | Complete repeated/spatial evidence; authenticate historical settings |
| R1-C2 | Architectural accuracy; must correct; pooling shared with R2 | 10 | 4 / 4 | 8 / 8 | Execute pooling/selective controls; verify historical model identity |
| R1-C3 | Reproducibility and variant identity; must correct | 9 | 0 / 2 | 6 / 7 | Recover NA configuration or replace that historical comparison |
| R1-C4 | Efficiency claim scope; must correct | 9 | 0 / 4 | 8 / 8 | R2-C3 still needs full comparative measurements |
| R1-C5 | Numeric correction and figure readability; must correct | 5 | 0 / 5 | 8 / 8 | Final rendered QA and re-audit replacement numbers |
| R2-C1 | Spatial validity; strongly recommended experiment treated as essential | 10 | 0 / 1 | 4 / 4 | Complete HC/HH disjoint comparisons for five repetitions |
| R2-C2 | State-space identity; must correct | 10 | 0 / 4 | 9 / 9 | Historical identity provenance remains separate |
| R2-C3 | Required profiling evidence | 9 | 0 / 1 | 3 / 3 | Full operation coverage and comparable GPU measurements |
| R2-C4 | Required repeated-run reliability | 9 | 0 / 0 | 3 / 3 | Execute five pairs; mean/sample SD; exact paired statistics |
| R2-C5 | Representative-token correctness; must correct | 10 | 0 / 1 | 9 / 9 | Preserved operation documented; no target-center claim |
| R2-C6 | Controlled novelty evidence; recommended and adopted | 9 | 0 / 3 | 3 / 3 | Seven controlled variants and external selective-block parity |
| R2-C7 | Additional generalization protocol; adopted spatial option | 9 | 0 / 1 | 4 / 4 | Spatial results; cross-scene transfer remains outside evidence |
| R2-C8 | Reproducibility; must supply | 9 | 0 / 2 | 6 / 6 | Authenticate baselines; complete archive and verified public release |
| R2-C9 | Interpretation; must correct | 7 | 0 / 4 | 7 / 7 | QUH paired FEB check and per-class results |

## Verified red flags and changes

1. **The present `main.py` is not executable.** It contains committed Git
   conflict markers. The snapshot records this state. Its incoming `initial_code`
   side parses and supplies the isolated reference classes; the untouched main
   file is not invoked by the revision runner. Matching this recovered model to
   every archived result remains unverified.
   A parameter-count probe confirms a concrete mismatch: recovered HC 52,760
   versus reported 37,540; HH 53,150 versus 37,930; TH 52,890 versus 37,670;
   PI 52,370 versus 37,150; QUH 52,110 versus 36,890. The difference is 15,220
   for every class count. `revision/model_identity.json` blocks production runs
   until the source/checkpoint/configuration is authenticated. Do not tune grid
   counts or widths simply to make the total coincide.
2. **The representative token is not the target-pixel spectrum.** For p=12,
   the actual target is local (6,6); a pixel-preserving row-major token index
   would be 78. The code selects spatial-view token 6. The transpose to
   [B,W,C,H] followed by reshape to [B,H*W,C] mixes rows and PCA components
   within a token. Simply changing 6 to 78 would not repair that mapping.
   The manuscript now describes it faithfully and preserves the computation.
3. **Fusion is indexwise, not pixel-aligned.** Equal indices in the two tensor
   views need not correspond to the same physical pixel. No semantic alignment
   claim is retained. The token projections/gates are independent; FSSA is shared.
4. **The recurrence is affine and non-selective.** Two biased dense maps are
   added, with fixed learned matrices, no recurrent activation, and a final-state
   classifier. There are no input-dependent Mamba selection parameters. The
   architecture drawing now follows the actual operations, including the absence
   of post-recurrence normalization. The framework name is retained as an identifier.
5. **Historical PCA fits held-out scene spectra.** `utils.DLMethod` fits whitened
   PCA to the entire flattened cube before center splitting. Those results are
   disclosed as transductive within-scene observations. New runs fit only training
   centers. Differences from historical tables cannot be attributed solely to
   spatial splitting because preprocessing also changes.
6. **Historical split seeds do not follow the training seed.** The split default
   is 345. The new matrix explicitly connects split/training pairs (345,0) through
   (349,4). Splits are shared across methods within a repetition.
7. **Validation does not select a checkpoint in the inspected path.** The script
   monitors validation but predicts from the last epoch. The revision states this
   explicitly; it does not falsely describe a validation-best selection rule.
8. **The optimizer decay is not weight decay.** Legacy Adam's `decay=1e-6` is a
   learning-rate decay argument. Active spline projections have a separate L2
   regularizer of 0.001. The manuscript now distinguishes these.
9. **Old FLOPs totals are inadequate.** The routine omits QK/AV and repeated token
   costs and can double-count recursive layers. New analytic accounting includes
   both attention calls and all scan positions, but is explicitly only a major
   matrix/convolution operation total. Complete FLOPs remain open.
10. **Parameter compactness does not imply runtime superiority.** Some listed
    Mamba-labeled models have fewer parameters. Quadratic attention remains.
    Training-ratio timing is a within-model observation, not a model benchmark.
11. **NA versus without-FSSA is unexplained.** HC has 77.53 versus 85.79 OA.
    The draft does not invent a reason. The new without-FSSA variant bypasses
    attention while retaining the documented frontend and recurrence.
12. **The local initial manuscript is not an authenticated original submission.**
    It contains two active state-space subsections with repeated labels, including
    revision text. Per the user's decision it is the comparison baseline; the
    original remains frozen and a full diff records deletions as well as additions.
13. **Numerical and causal overstatements are removed.** HH's OA difference is
    2.80 percentage points. QUH without-FEB improves OA by 0.09 points and kappa
    by 0.11 points but lowers AA by 0.37 points in one historical run. This does
    not prove a stable benefit/harm or explain its physical cause.
14. **Original response boilerplate claims completion prematurely.** Replaced
    claims of final approval and completed uploads with a clearly marked working
    draft. Every evidence-dependent concern has a visible pending-action notice.
15. **Baseline identity is not established by filenames.** Some local Mamba-labeled
    implementations also contain dense recurrences; `models/Mamba_model.py` has an
    invalid `re` layer import. SSFTTKAN and HybridKAN importable sources were not
    found. Registry entries remain unverified rather than substituting candidates.

## Exact manuscript changes and reviewer traceability

| Manuscript location | Revision | Comments |
|---|---|---|
| Abstract and contributions | Qualify novelty, historical evidence and efficiency; flag missing results | R1-C2/C4, R2-C2/C6/C7 |
| Fig. architecture | New code-grounded diagram, fixed representative descriptor and affine scan | R1-C2/C3, R2-C2/C5 |
| II.A Token generation (`sec:token_details`) | Exact permutation/reshape and coordinate equations; indexwise fusion | R1-C3, R2-C5 |
| II.B FastKAN (`sec:fastkan`) | Eight centers, trainability, initial interval/spacing, active initialization/regularization | R1-C3, R2-C8 |
| II.C FEB (`sec:feb`) | Index 6 versus target (6,6)/78; independent gates | R1-C3, R2-C5 |
| II.D FSSA (`sec:fssa`) | Shared branch attention, distinct Q/K/V, 32 channels, eight heads, dropout | R1-C3, R2-C8 |
| II.E recurrence (`sec:ssm`) | Affine update, both biases, final-state classifier, complexity limits | R1-C2/C4, R2-C2/C3 |
| III (`sec:experiments`) | Historical random sampling, whitening, full-scene fit, ratio semantics, final epoch | R1-C1, R2-C1/C4/C8 |
| IV results and isolated historical tables | Within-scene single-run labels, corrected HH difference, limited visual claims | R1-C4/C5, R2-C7 |
| V historical attention/module interpretation | Unresolved NA provenance; metric-dependent FEB effect without invented mechanism | R1-C3, R2-C6/C9 |
| V revision protocol (`sec:revision_protocol`) | Complete experiment, sampling, profiling, statistics and release specification | All experimental concerns |
| VI conclusion/limitations (`sec:limitations`) | Claims bounded by evidence; cross-scene transfer not established | R1-C1/C2/C4, R2-C1/C4/C6/C7 |

Final PDF page numbers must be recorded after completed results settle pagination.
Do not cite a draft page number as final. The response uses stable section names.

## Experiments and logical justification

| Study | Runs | Logical question | Required controls |
|---|---:|---|---|
| Main random comparison: 9 methods x 5 scenes x 5 repetitions | 225 | Does comparative performance persist over partition/training variation? | Same split/preprocessing and declared training schedule; authenticated model identities |
| Spatial comparison: 9 methods x HC/HH x 5 | 90 | Does comparative performance persist without shared patch pixels? | Region assignment before extraction; complete support separation; training-only PCA |
| Seven variants x HC/HH x random/disjoint x 5 | 140 | Which projection/gating/aggregation choices contribute under controlled changes? | Same frontend/interfaces and seeds; report parameter differences |
| QUH without-FEB x 5 | 5 | Does the single-run FEB exception persist, and which classes change? | Pair with main QUH run; inspect class-frequency-weighted OA and equal-class AA |
| Profiling, separately from training | No extra full fits required | Is compactness accompanied by practical speed/memory benefits? | Idle same GPU, precision, shapes, warm-up, synchronization, operation coverage |

Seven variants: normalized linear Q/K/V; normalized GELU-MLP Q/K/V (hidden 128);
MLP FEB gates; without FEB; without FSSA; mean pooling; selective-Mamba aggregation.
The last follows Mamba-1 equations with model/state width 64, convolution width 4,
expansion 2 and final-position readout. Its scan is tested against a NumPy
recurrence but external full-block parity remains pending. Its sequential TF
runtime must not be equated with the official fused CUDA implementation.

The pilot is three training/validation epochs and never predicts the test set.
Fix total GPU-hours after observing pilot costs, before test results. Hardware
access and a submission deadline are not currently available to the local runner.
All completed scheduled outcomes must be retained; favorable-outcome selection
does not become acceptable when accuracy decreases.

## Spatial feasibility ruling and coverage

The original plan's 64-pixel blocks were tested using only ground-truth geometry
and sample counts. HC seed 345 is feasible: train 12,882; validation 12,882;
test 106,557. HH at 64, 48 and 32 is infeasible under the complete-class/5%-quota
constraints. At 24 it is feasible: train 19,345; validation 19,345; test 68,071
for seed 345. Exact masks/counts for every repetition must be archived separately.

**Ruling:** fix HC=64, HH=24 before any model evaluation. This is a documented
departure from the fixed-64 plan, justified by a failed feasibility prerequisite,
not by accuracy. Smaller HH blocks exclude many boundary centers and produce a
different geographic scale and test population. Report these differences; do
not describe either protocol as cross-scene transfer or statistical independence
of all scene pixels. The hard guarantee is zero shared patch support.

## Statistical interpretation

Report all five values plus mean and sample SD (`ddof=1`). Seed pairs change
partition and initialization/shuffling together, measuring pipeline variation.
Methods are paired by split hash. Repetitions share an underlying scene and do
not establish a population-level confidence interval for new scenes.

The exact two-sided sign-flip test has 32 assignments for five nonzero pairs;
its minimum p-value is 2/32=0.0625. Holm correction cannot make that smaller.
Consequently **this test cannot establish significance at 0.05 with five pairs**.
Report the result honestly, alongside effects and variability. Do not switch to
a more favorable test after seeing scores or call nonsignificance equivalence.

Predeclared primary families: random main comparisons (40), spatial main (16).
Exploratory ablations use separate families: random (15) and spatial (14).
Missing planned comparisons count as p=1 for family size accounting and remain
explicitly missing. The collector returns incomplete until all 460 runs exist.

## Reproducibility archive and release gates

Store dataset hashes/variable keys; exact center IDs; block assignment; excluded
counts; fitted PCA mean/components/variance; matrix/configuration; code hashes;
framework/build/GPU environment; checkpoint; epoch history; timings; predictions
and confusion matrices. Do not publish proprietary credentials or raw private data.
The benchmark data are not added to Git. Publication/release is not performed
by this local task and no public URL is invented.

## Current implementation and verification status

- Completed: separate revision source package, frozen reference copies, response
  draft, highlighted/clean drafts, per-comment evidence ledger, corrected figures,
  deterministic matrix, split/PCA tooling, preserved model and seven ablations,
  single-run/pilot entry points, result aggregation and profile entry points.
- Verified locally: synthetic split and preprocessing invariants; real HC/HH
  label-only feasibility; model output parity after identical-weight copy;
  all seven variants' finite gradients; selective scan against a NumPy recurrence;
  aggregate summary/paired-test behavior. See `revision/verification.json` for the
  final command outputs and document QA outcome after verification completes.
- Open: remote training, authenticated baseline adapters, historical-run model
  provenance, external selective-block parity, complete FLOPs accounting,
  comparative profiles, completed-result prose, public release and final pages.

### Final local verification (2026-10-26)

- **22/22 tests pass**, including the compiled TensorFlow training-step smoke
  test, exact AST preservation against the frozen source, all seven ablations'
  finite gradients, cache/configuration rejection, split/label authentication,
  changed-model rejection, and the production reference-authentication guard.
- Source checks pass for all 14 response/action pairs, active equation/section
  references, citation keys, required assets, and all 31 unchanged originals.
- Tectonic 0.17.0 successfully compiled the response (5 pages), highlighted
  manuscript (14 pages), and clean manuscript (13 pages). Every rendered page
  was reviewed. Automated PDF checks found no unresolved references, replacement
  glyphs, or words outside page bounds. Inherited footer-box/font warnings are
  recorded; compilation success is not a claim of a warning-free TeX log.
- The independent review identified four important tooling defects, now fixed:
  baseline operation-counter dispatch, split/artifact/ground-truth identity,
  cached protocol/configuration identity, and baseline/model source provenance.
  Regression tests cover their failure modes.
- **No new benchmark training run or GPU profile has been completed.** The
  460-run matrix remains pending. The confirmed 15,220-parameter model mismatch
  is a production gate, not merely an advisory note. Server access and verified
  baseline sources are also still required. This package is a working revision,
  not an acceptance-ready or experiment-complete submission.

## Source basis

- Local executable definitions and original reviewer comments, frozen under
  `revision/reference/` with hashes in its `manifest.json`.
- [Mamba paper](https://arxiv.org/abs/2312.00752): input-dependent selection differs
  from fixed affine recurrence.
- [Official Mamba-1 block](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
  and [reference selective scan](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py).
  Accessed 2026-10-25; pin a commit during external parity verification.
- [IEEE Access peer-review process](https://ieeeaccess.ieee.org/authors/stages-of-peer-review/):
  point-by-point reviewer treatment; acceptance remains an editorial decision.

## Execution rulings

1. Existing detached checkout is used in place; additions are isolated under
   `revision/` and revision-specific paper assets. A separate worktree would omit
   the ignored manuscript, datasets and model files. Originals are frozen and
   unchanged. No branch switch, commit, merge, or push is performed.
2. Incoming conflict-side recovery is a candidate reference, not authentication
   of archived runs. This uncertainty is visible in the response and manuscript.
3. HH uses 24-pixel blocks after label-only infeasibility at larger fixed widths;
   the coverage/geographic-scale limitation must be preserved in reporting.
4. The original font packages conflict, and the old class hardcodes pdfTeX
   spot-color primitives. A revision-specific compatibility class and coherent
   Times stack support compilation without modifying the original class.
5. Missing evidence is kept visible. A polished draft is not assigned a false
   completion score, and the generated clean manuscript retains pending notices.
