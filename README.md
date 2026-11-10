# eeg_calibration_mechanism

What does calibration fix? Neurotech@Berkeley, Project A.

A decoder trained on other people loses most of its accuracy on a new person. A per-user channel map (64 numbers at 8
channels) recovers part of it ([eeg_subject_alignment](https://github.com/navyers21/eeg_subject_alignment)). This repo
asks what is lost across people and what the map brings back, using Karan Singh's probe and erasure toolkit, and then
tests calibrating only the part that breaks.

## Layout

| Path | What it is |
| --- | --- |
| `mi/` | Karan's model-agnostic toolkit, unchanged: image labels, grouped-CV probes, LEACE + norm-matched random erasure, retrieval metrics (top-1, category, within-category), bootstrap over concepts |
| `interp/ubp_adapter.py` | Connects `mi/` to our UBP models. Modes: `within`, `f0`, `f0_map`, `sulora_map`, `sub_*`; eval sets `held` (200 held-out concepts x 10 images) and `test` (80-rep, analysis only) |
| `src/eegalign/` | UBP backend, cross-subject sources, calibration adapters (from eeg_subject_alignment) plus `subspace.py` (new), `props.py` (cached image properties), within-subject decoders in `source.py` |
| `train_within.py` | Within-subject decoder per subject: same architecture, targets and recipe as f0, own data only |
| `run_interp.py` | Runs `mi/pipeline.py` for one subject on every model mode; outputs to Drive `interp/` |
| `gap_analysis.py` | Per measure: lost = within - f0, restored = mode - f0, fraction restored, p-values across subjects |
| `calibrate.py` | Calibration runs (subspace phase), one CSV row each, resumable |
| `scripts/sweep.py` | Phases: `props`, `within`, `interp`, `gap`, `subspace`, `summary` |
| `scripts/download_data.py`, `scripts/build_props.py` | Data and config; one-time low-level property cache |
| `notebooks/colab.ipynb` | Thin runner |

## Differences from Karan's NeuroBridge run

| | Karan (NeuroBridge) | Here (UBP) |
| --- | --- | --- |
| Models | Within-subject only | Within-subject, cross-subject f0, f0 + map, SuLoRA-style map |
| Image features | NeuroBridge release (mismatched, cosine 0.68) | Extracted from the images with OpenCLIP RN50 |
| Held-out set | 165 random concepts | The 200 held-out concepts every calibration run uses |
| Semantic labels | k-means of release features | k-means of correctly extracted features (THINGS categories are a planned swap) |
| Montage | All 63 | Posterior-17 and occipital-8 |

## Mechanism-guided calibration
The subspace is defined without the new user's labels: the LEACE-style whitened subspace that carries the 6 low-level
properties (`sub_low`) or the 20 semantic clusters (`sub_sem`) in f0's first-layer activations on the SOURCE subjects'
recordings; `sub_rand` is a random control. A new user learns only a 6 x 6 matrix inside it (36 numbers).

## Go / no-go (fixed before running)
1. `gap`: the fraction restored by the map must differ clearly between low-level (within-category retrieval,
   low-level probes, low-level reliance) and semantic measures. If the map restores everything about equally: no
   subspace story.
2. `subspace`: `sub_low` must match or beat the SuLoRA-style map at 25-50 images and beat `sub_rand`.

Targets 2-10 (subject 1 tuned the calibration recipe). Caveat: 8- and 17-channel data use UBP's 63-channel whitening.
