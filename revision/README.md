# FKANMamba evidence revision

This package prepares the approved five-repetition study and generates honest
revision drafts. **No new benchmark result is supplied by the package itself.**
The response and manuscript remain working drafts until the evidence gaps in
`paper/revision_audit.md` are closed.

## What is available

- Frozen copies and SHA-256 hashes of 31 original source/manuscript files.
- An importable reference model extracted from the incoming `initial_code` side
  of the committed conflict in `main.py`; the original file is untouched.
- The preserved forward operation and seven isolated, executable ablations.
- A 460-run matrix, five paired seed configurations, class-stratified random
  splits, support-disjoint splits, training-center-only whitened PCA, lazy patch
  batches, final-epoch training, and protected per-run output directories.
- Prediction-based metrics, complete-five-repetition summaries, exact paired
  tests and Holm correction, and synchronized GPU profiling commands.
- A response for every concern, highlighted/clean manuscript drafts, corrected
  operation diagram, readable classification-map layouts, and a change diff.

## Blocking evidence and implementation gaps

1. Server SSH access has not been provided. This local TensorFlow environment
   has no visible GPU. The full study has not been launched.
2. None of the eight historical baseline implementations is authenticated.
   `adapters.json` lists candidates and reasons. SSFTTKAN/HybridKAN importable
   implementations were not found. The runner refuses an unverified adapter;
   it never silently substitutes a similarly named local model.
3. The recovered model is internally tested, but has **52,760 HC parameters
   versus the paper's 37,540**. All five datasets show the same 15,220 difference.
   `model_identity.json` records the mismatch and blocks production training.
   Authenticate the original source/checkpoint/configuration; do not guess layer
   widths or grid counts merely to force a matching total. The explicit
   `--allow-unverified-reference` flag permits diagnostic pilots only.
4. The selective-Mamba control has a tested selective-scan recurrence, but
   external parity with the official Mamba-1 block is not yet established.
   It uses a sequential TensorFlow scan, not the fused official CUDA kernel.
5. `flops.py` provides **major matrix/convolution MACs and their 2-MAC FLOP
   equivalent**, with excluded operations listed. It is deliberately marked
   `complete_flops=false`. Complete FLOPs/operation coverage and authenticated
   baseline profiling remain outstanding. Do not compare this partial total
   against another method's full profiler total.
6. The 64-pixel block plan was infeasible for HH under the required class and
   sample constraints. Label-only checks found 48 and 32 infeasible too; HH is
   fixed at 24 and HC at 64 before training. This changes the retained coverage
   and must be reported. No test accuracy entered this choice.

## Environment

Use Linux with TensorFlow/Keras 2.15.1 for the RTX 4090 server. The historical
`req.txt` is a list of shell commands, not a pip requirements file. NumPy,
SciPy (including `scipy.optimize.milp`), scikit-learn, h5py, matplotlib, and
Pillow are needed. Record actual Python/driver/CUDA/cuDNN versions using
`preflight`; do not present a suggested environment as the executed one.
Training refuses to fall back to CPU silently; `--allow-cpu` is reserved for
explicit local diagnostic pilots. GPU determinism is requested explicitly. Some kernels/platforms may refuse
deterministic execution; retain and investigate that failure rather than
silently disabling the setting. MATLAB v7.3/HDF5 inputs require a separately
validated loader; the current loader accepts standard SciPy-readable MAT files.

## Server commands

Run from the repository root. The examples assume the server's dataset folder
is `/data/hsi`; replace that path with the actual folder. Dataset file names
and MATLAB variable keys are defined in `runner.FILES`.

```bash
python -m unittest discover -s revision -p 'test_*.py' -v
python -m revision.runner matrix --artifacts revision_artifacts
python -m revision.runner preflight --data-root /data/hsi --artifacts revision_artifacts

# First run a training-only pilot; it does not predict the test partition.
CUDA_VISIBLE_DEVICES=0 python -m revision.runner pilot \
  --run-id HC__random__FKANMamba__r0 --data-root /data/hsi --pilot-epochs 3 \
  --allow-unverified-reference

# Prepare partitions/PCA serially before parallel jobs sharing a split.
python -m revision.runner prepare \
  --run-id HC__disjoint__FKANMamba__r0 --data-root /data/hsi

# Actual final-epoch run (50 epochs, five separate repetitions in the matrix).
CUDA_VISIBLE_DEVICES=0 python -m revision.runner train \
  --run-id HC__random__FKANMamba__r0 --data-root /data/hsi

# Use an otherwise idle GPU and a fresh process for profiling.
CUDA_VISIBLE_DEVICES=0 python -m revision.runner profile \
  --run-id HC__random__FKANMamba__r0 \
  --weights revision_artifacts/runs/HC__random__FKANMamba__r0/final.weights.h5

python -m revision.results --artifacts revision_artifacts

# After authenticated adapters and pilot budgeting, inspect the launch first.
python -m revision.launch --data-root /data/hsi
# Example only: choose the actual budget from measured pilot costs.
# python -m revision.launch --data-root /data/hsi --gpu-hours YOUR_BUDGET --execute
```

The summary command returns exit code 2 when the 460-run study is incomplete
or invalid. Complete subgroups can still be inspected, but an incomplete study
is never labeled complete. It recalculates metrics from saved predictions.
The two-sided exact test with five pairs has minimum p=0.0625, even before Holm
correction; report limited power and avoid claims of significant superiority.

Prepare each of the 35 unique dataset/protocol/seed partitions once. Do not run
two simultaneous `prepare` commands on the same split. Use one training job
per GPU. Do not profile while another job occupies that GPU. Output directories
are exclusively created: a failed or unfavorable run is never overwritten.
Investigate failed runs, preserve their artifacts, and record any authorized
replacement attempt separately before adding it to the final evidence ledger.

## Authenticating a baseline adapter

Provide a small importable factory with this interface:

```python
def build(*, patch_size: int, components: int, classes: int):
    # Return an uncompiled tf.keras.Model accepting float32 [B,12,12,15]
    # and returning [B,classes] probabilities. No data loading or training here.
    ...
```

It must encapsulate the baseline's declared input transformation, forward
architecture and dropout, and avoid global dataset/GPU changes. Compare it
with the cited authors' implementation and record source URL/revision,
configuration and parity checks. In `adapters.json`, set `factory` to
`module.path:build`, supply the module's `source_sha256`, and set `verified`
only after that verification. Hash imported dependencies as part of the
experiment archive too. The current common runner uses the frozen common
training schedule; authenticate and report this harmonized protocol rather
than claiming it reproduces baseline authors' independently tuned schedules.

## Rebuilding documents and figures

```bash
python -m revision.build_figures
python -m revision.build_documents
python -m revision.validate_documents
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error repose_filled.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error highlighted_filled.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error clean_filled.tex
```

The supplied 2014 IEEE Access class uses pdfTeX-only spot-color primitives.
The revision-specific class retains its pdfTeX branch and adds a CMYK fallback
for XeTeX/Tectonic preview. The conflicting original newtx/mathptmx font stacks
are reduced to the standard Times stack. The original class is unchanged.
Class/BibTeX/author-image assets needed for compilation are copied into
`paper/revision_assets/support/`. Rebuilding historical map panels additionally
requires the original images under `paper/Fig/`.

Classification maps are cropped into their existing panels and re-laid out;
predictions/class colors are not recomputed or altered. Source hashes and crop
boxes are in `paper/revision_assets/figures/provenance.json`. The architecture
diagram is regenerated from the actual code operations.

The generated response filename is deliberately **`repose_filled.tex`**, matching
the requested spelling. `clean_filled.tex` shares the highlighted manuscript's
scientific content. Red pending-evidence notices are present in both versions
until experiments are completed; disabling yellow does not make a draft final.

## Implementation record

- Ruling: work in the existing detached checkout and isolate additions under
  `revision/` rather than creating a worktree lacking ignored paper/data/model
  assets. No commit, branch switch, push, or historical-source overwrite.
- Ruling: recover the incoming conflict side as a candidate reference and freeze
  both the conflicted source and resolved copy; do not assert historical identity.
- Ruling: HH block size 24 is chosen solely for label-only feasibility, prior
  to training. Costs: reduced coverage and a different spatial scale from HC.
- Ruling: mark baseline identity, selective-block external parity, full FLOPs,
  and unexecuted experiments as open requirements rather than inventing evidence.
- Ruling: compile with an engine-compatible class copy and one coherent font
  stack; original source/class files remain frozen for comparison.
