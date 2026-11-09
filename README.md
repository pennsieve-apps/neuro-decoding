# neuro-decoding

**Built on published decoding methods and scikit-learn, not developed by Pennsieve.** The
cross-validation and decoder ladder follow the best-practice guide of Joshua Glaser, Ari Benjamin,
Raeed Chowdhury, Matthew Perich, Lee Miller and Konrad Kording ([Glaser et al., *eNeuro* 2020](https://doi.org/10.1523/ENEURO.0506-19.2020));
the Bayesian position decoder is the population reconstruction of Kechen Zhang, Iris Ginzburg,
Bruce McNaughton and Terrence Sejnowski ([Zhang et al., *J Neurophysiol* 1998](https://doi.org/10.1152/jn.1998.79.2.1017)), as in
[pynapple](https://github.com/pynapple-org/pynapple)'s `decode_1d`; held-out-session decoding
through aligned latent dynamics follows Juan Gallego, Matthew Perich, Raeed Chowdhury, Sara Solla and
Lee Miller (Miller lab, Northwestern University; [Gallego et al., *Nat Neurosci* 2020](https://doi.org/10.1038/s41593-019-0555-4)); kNN
decoding from CEBRA embeddings follows Steffen Schneider, Jin Hwa Lee and Mackenzie Mathis (EPFL;
[Schneider et al., *Nature* 2023](https://doi.org/10.1038/s41586-023-06031-6)); the pseudo-session
nulls and null-corrected, Fisher-combined region statistics for trial variables follow the
International Brain Laboratory ([Findling, Hubert et al., *Nature* 2025](https://doi.org/10.1038/s41586-025-09226-1);
[IBL, *Nature* 2025](https://doi.org/10.1038/s41586-025-09235-0)); communication subspaces between
brain regions (reduced-rank regression) follow João Semedo, Amin Zandvakili, Christian Machens,
Byron Yu and Adam Kohn ([Semedo et al., *Neuron* 2019](https://doi.org/10.1016/j.neuron.2019.01.026))
and Camden MacDowell, Timothy Buschman and colleagues ([MacDowell et al., *Nat Commun* 2025](https://doi.org/10.1038/s41467-025-58698-2)). Ridge, MLP, logistic
regression, kNN and PCA come from [scikit-learn](https://scikit-learn.org) (Pedregosa et al. 2011).
Pennsieve packages these methods as a workflow step and did not develop them; please cite the
original papers (below). The demo data are from Andres Grosmark and György Buzsáki (NYU;
DANDI:000044) and the International Brain Laboratory (DANDI:000409).

The step after `neuro-align` in the **Neural Population Dynamics & Decoding** workflow:

- `neuro-align → {neuro-encoding, neuro-embedding, neuro-decoding} → neuro-report`

**For a neuroscientist.** How much does the recorded population know about what the animal is
doing? This app trains decoders that read behaviour (position, running speed, wheel velocity,
pupil, the animal's choice, which side the stimulus was on...) from the activity of the recorded
neurons, and tests them on parts of the session they were not trained on. Every score is shown next
to what you would get by chance with the same data (behaviour shifted in time against the activity,
or trial labels shuffled), so a high number that is not better than chance is visible as such. It
also shows how decoding grows with the number of neurons, which brain regions carry which
variable (IBL recordings span many regions), how strongly the recorded regions predict each
other's activity and through how many dimensions ("communication subspaces"), and whether a
decoder learned in one animal works in another. The headline figure is the decoded trace laid over the true one.

**For a computational analyst.** Continuous variables, per session: features are square-rooted
spike counts in `DECODE_BIN_S` bins, Gaussian-smoothed (`SMOOTH_S`), with `HISTORY_BINS` bins of
history (BrainWideBench, Andre et al. 2026, finds a history window the cheapest large gain for
wheel decoding); to keep the feature count manageable the lags are grouped on a log scale, as a
raised-cosine history basis would (10 bins: lag 0, 1, 2-3, 4-7, 8-9, each group averaged, so 5
features per neuron), z-scored on the training fold
(SD floored, z clipped at ±10 so that a unit lost or found mid-session cannot blow up a fold);
`CV_FOLDS` contiguous blocks with `CV_GAP_S` removed from the training set on both sides of each
test block. Decoders: ridge (alpha from a log grid by inner 3-fold blocked CV; every alpha from one
eigendecomposition of the Gram matrix), `MLPRegressor` (64-32, L2 penalty 1.0, early stopping,
predictions clipped to the training range), and for position variables of spiking data a Poisson naive-Bayes decoder
(occupancy-normalised tuning curves from the training fold, occupancy prior, posterior mean, counts
in a 0.5 s window). Score: mean of per-fold R² (also Pearson r and median absolute error in the
variable's unit). Null: `N_SHUFFLES` circular shifts of the behaviour by at least `MIN_SHIFT_S`,
re-fitting ridge with the alpha of the real fit on the same folds; p = (1 + #null ≥ real) / (1 + N),
z against the null. Trial variables: spike counts in `TRIAL_WINDOW_S` around `TRIAL_ALIGN`
(stimulus onset by default), logistic regression (C by inner stratified CV, balanced class
weights), contiguous trial folds, balanced accuracy. Their null, for a session with the IBL task
structure (a 0.2 / 0.5 / 0.8 prior and a stimulus side), is IBL's **pseudo-session** null: label
sequences drawn from the task's generative process (90 unbiased trials, then blocks alternating
0.2 / 0.8 with lengths from a truncated exponential of mean 60 trials, the stimulus side drawn from
the block prior, contrasts resampled from the session, correct choices with the session's
accuracy per contrast), decoded from the real activity with the same pipeline; other sessions use
trial-label permutations. Every score gets a **null-corrected score** (score minus the null mean;
for R², minus max(null mean, 0)), and `regions.csv` combines each region's p-values over sessions
with Fisher's method. Neuron-dropping:
random subsets of 1 to N neurons (8 sizes, `DROP_REPEATS` each), fixed alpha / C. Regions: every
brain region with ≥ `MIN_REGION_NEURONS` neurons decoded alone, with its own null (≤ 100 shuffles).
Communication subspaces: for every pair of regions with ≥ `MIN_UNITS_PER_REGION` units, reduced-rank
regression of held-out target units on source units (see Method 5).

## Ports

| Port | Direction | Content |
|---|---|---|
| `aligned` | input (subfolder, required) | neuro-align's output: `neuro-align/<session_id>/aligned.npz`, `events.parquet`, `manifest.json` (`spikes.npz` is not needed). |
| `embedding` | input (subfolder, optional) | neuro-embedding's output: `neuro-embedding/<session_id>/<session_id>_embeddings.npz`. Adds kNN decoders. |
| `decoding` | output | `neuro-decoding/...` (below) |

Without links everything may sit side by side under `INPUT_DIR`; files are found by name.

## Method

1. **Variables.** Continuous: every column of neuro-align's `Y` that is valid in at least half of
   the epoch and not constant (`VARIABLES=auto`; `wheel_position`, a cumulative angle that drifts
   over a session, only when named; angles become sine and cosine). Trial variables
   (`TRIAL_VARIABLES=auto`): every column of the trials table with 2 to 6 distinct values (IBL:
   `gabor_stimulus_side`, `mouse_wheel_choice`, `is_mouse_rewarded`, `probability_left`,
   `block_type`, `gabor_stimulus_contrast`; other datasets: `choice`, `feedback_type`, ...), plus
   `stimulus_side` from `contrast_left` / `contrast_right` when present; classes with fewer than 5
   trials are dropped.
2. **Within-session decoding** with blocked CV, the decoders and nulls above, per region too.
3. **Held-out session** (two or more sessions that each decode a variable significantly): every
   session's features are reduced to `HELDOUT_DIMS` principal components; the other sessions'
   components are rotated onto a reference session (orthogonal Procrustes on the mean latent per
   bin of the variable × running direction), a ridge decoder is trained on them, the held-out
   session is rotated with the first half of its own data (calibration, labels used only for the
   alignment) and decoded on its second half. Reported next to its own decoder (trained on the first
   half) and to no alignment.
4. **CEBRA kNN** (when `embedding` is linked; an optional comparison, not the headline decoder — on
   IBL data CEBRA-kNN does about as well as a linear decoder for the wheel and near chance for the
   choice; [Andre et al. 2026, BrainWideBench](https://doi.org/10.48550/arXiv.2609.22064)): kNN regression (cosine, `KNN_K` neighbours) from
   the CEBRA-Time embedding (label-free, so the same blocked folds apply) and from the CEBRA-Behavior
   embedding on the bins neuro-embedding held out of its training (its `cebra_behavior_train_mask`;
   one split, so no leakage of labels into the embedding).
5. **Communication subspaces** (units with brain regions, at least two regions with ≥
   `MIN_UNITS_PER_REGION` units; otherwise skipped with the reason logged): each target region's
   units are split at random into halves; the second half is predicted from the first (within
   region) and from each source region subsampled to the same number of units (between regions),
   on the epoch's bins, with the same blocked folds, after removing each unit's 60-s running mean
   (slow drifts; Semedo et al. used trial residuals). Each prediction is a full-rank ridge model and
   its reduced-rank versions (ridge coefficients projected on the top r axes of the fitted
   predictions, r = 1 .. `RRR_MAX_RANK`); the optimal rank is the smallest r within one standard
   error of the full model (Semedo et al. 2019). Plain numpy.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `VARIABLES` | `auto` | Continuous variables (neuro-align names, comma-separated), `auto`, or `none`. |
| `TRIAL_VARIABLES` | `auto` | Trials-table columns to decode per trial, `auto`, or `none`. |
| `TRIAL_ALIGN` | `auto` | Column the trial window is aligned to (auto: stimulus onset, else trial start). |
| `TRIAL_WINDOW_S` | `-0.1,0.5` | Window (s) around `TRIAL_ALIGN` whose spikes are counted. |
| `DECODE_BIN_S` | `0.1` | Bin (s) of continuous decoding; 0 keeps neuro-align's bins. |
| `SMOOTH_S` | `0.05` | Gaussian SD (s) of the feature smoothing. |
| `HISTORY_BINS` | `10` | Bins of history in the ridge / MLP features (current bin and the 9 before: 1 s at 0.1 s bins). |
| `CV_FOLDS` | `5` | Contiguous folds. |
| `CV_GAP_S` | `2.0` | Training data dropped (s) on each side of a test block. |
| `N_SHUFFLES` | `200` | Null size (circular shifts / label shuffles); per-region nulls ≤ 100; 0 = none. |
| `MIN_SHIFT_S` | `20` | Smallest circular shift (s). |
| `DECODERS` | `ridge,mlp,bayes,logistic,knn,rrr` | Analyses to run (ridge always runs; `rrr` = communication subspaces). |
| `N_POS_BINS` | `40` | Position bins of the Bayesian decoder. |
| `DROP_REPEATS` | `10` | Random subsets per size in the neuron-dropping curve. |
| `MIN_REGION_NEURONS` | `10` | Smallest region decoded alone. |
| `MIN_UNITS_PER_REGION` | `10` | Smallest region in the communication-subspace analysis. |
| `RRR_MAX_RANK` | `10` | Largest rank tried by reduced-rank regression. |
| `HELDOUT_DIMS` | `10` | Latent dimensions of held-out-session decoding. |
| `KNN_K` | `25` | Neighbours of the kNN decoder. |
| `TRACE_WINDOW_S` | `60` | Window (s) of the decoded-vs-true figure (the start of the middle fold). |
| `SEED` | `42` | Seed of shuffles, subsets and the MLP. |
| `THREADS` | `0` | BLAS threads; 0 = the container's CPU quota. |

Exit codes: `0` ok, `1` no `aligned.npz` or every session failed, `2` bad parameters. A session
that fails (e.g. no decodable variable) is logged and skipped.

## Outputs

`neuro-decoding/` in the `decoding` port:

- `decoding.csv`: one row per session × variable × decoder × region (and per held-out session):
  `session_id, subject_id, condition, group, variable, kind` (continuous / categorical), `unit,
  decoder` (ridge, mlp, bayes, logistic, knn), `features` (activity, `cebra_time`,
  `cebra_behavior[...]`, PCA latents, or the trial window), `region` (`all` or a brain region),
  `n_neurons, n_samples` (bins or trials), `cv` (how it was cross-validated), `metric` (`r2` or
  `balanced_accuracy`), `score, score_sd` (across folds), `r` (Pearson), `median_abs_error`,
  `chance` (1 / classes), `null` (circular shift / pseudo-session / trial-label shuffle),
  `score_corrected` (score minus the null mean; R²: minus max(null mean, 0)), `null_mean, null_sd,
  null_p95, p_value, z, n_shuffles`.
- `regions.csv`: per variable × region × decoder over sessions: `n_sessions, n_neurons, mean_score,
  mean_score_corrected, fisher_p` (Fisher-combined p-value), `null`.
- `rrr.csv`: `session_id, source, target, kind` (within / between), `n_source, n_target, n_bins,
  optimal_rank, rrr_r2` (at the optimal rank), `full_r2` (full-rank ridge), `within_r2,
  within_rank` (the target predicted from its own other half); `<session_id>/<session_id>_rrr.png`.
- `dropping.csv`: `session_id, variable, n_neurons, mean, sd, repeats`.
- `heldout.csv` (`variable, session_id, train_sessions, reference, n_dims, n_conditions_matched,
  n_test, within_session, aligned, unaligned`) and `heldout_<variable>.png`.
- `<session_id>/<session_id>_decoded_traces.png` (decoded vs true; trial variables as decoded
  class per held-out trial), `<session_id>_null.png` (score vs null per variable and region),
  `<session_id>_dropping.png`, `summary.json`.
- `summary.json` (per session and variable: scores, nulls, p-values, regions; held-out sessions;
  figures with captions in report order), `provenance.json`.

## Runtime

Measured in Docker (linux/amd64 emulation on an M-series Mac, `--cpus 2 --memory 4g`, the builder
brief's Lambda recipe: read-only root, arbitrary user, only `/tmp` writable) on the demo data
(`data/work/<demo>/neuro-align/`), with the defaults, on a host **shared with many other builds
(load average 50-150)**, so wall times are pessimistic:

| Session | Analyses | Wall time | Peak memory (cgroup) |
|---|---|---|---|
| Grosmark `sub-Buddy` (58 units, run epoch 10,654 bins of 25 ms → 2,698 bins of 0.1 s; 6 continuous variables + lap direction; embedding linked) | ridge, MLP, Bayes, kNN on CEBRA, nulls, dropping | 94 s | 670 MB |
| IBL `sub-NR-0031` (186 units, 13 regions, 80,064 → 20,016 bins; 6 continuous + 7 trial variables) | everything, incl. 4 regions and communication subspaces | 2,104 s (continuous variables ~5 min each; trial variables ~30 s each; RRR 29 s) | 1,311 MB |
| Synthetic fixture (3 sessions) | everything | 120-210 s | 0.5 GB |

The time goes into the nulls (200 shifts per variable and 100 per region; each is one matrix
product thanks to a per-fold decomposition) and the MLP. On a typical session with few variables
the step fits Lambda's 15 minutes; a large IBL session with every pose variable does not, so
`timeoutSeconds` is 3600 on standard compute (`N_SHUFFLES=100` or naming `VARIABLES` halves it).

## Demo results

`data/work/<demo>/neuro-decoding/`:

- **Grosmark & Buzsáki, rat Buddy** (58 CA1 units, running epoch): 1-D track position decoded with
  R² **0.70** (ridge, 1 s of history; null 95th percentile -0.10, p = 0.005 with 200 circular
  shifts), **0.80** (MLP) and **0.72** (Bayesian decoder; median error **7.7 cm**; the source file's position variable spans about 2.5 m for the 1.6 m track, so errors are in its units); kNN on the CEBRA-Behavior embedding (held-out bins) 0.53, on CEBRA-Time 0.44.
  Running speed R² 0.45-0.47; lap direction (out / in) balanced accuracy **0.89** (chance 0.5,
  null p95 0.62). The neuron-dropping curve rises steadily (1 unit 0.01, 10 units 0.30, 32 units
  0.51, all 58 units 0.70): position is spread over the population and more units would help.
  One region only, so no communication subspaces.
- **IBL `sub-NR-0031`** (186 units): wheel velocity R² **0.17** with ridge and **0.40** with the
  MLP (null p95 -0.04); right / left paw speed 0.44 / 0.40, nose tip 0.28, pupil 0.29, licks 0.05
  (all p = 0.005). Trial variables in a -0.1 to 0.5 s window around the stimulus, against IBL
  pseudo-sessions: **choice** balanced accuracy **0.86** (pseudo-session null p95 0.55, corrected
  +0.36), **stimulus side 0.78** (null p95 0.57, corrected +0.30), reward 0.67 (corrected +0.17);
  the block prior (`probability_left`) is *not* decodable in this window (0.57 vs null p95 0.67,
  p = 0.40), which a label shuffle would have wrongly reported as significant (shuffle null p95
  0.38 in an earlier run). Per region (corrected scores): stimulus side in the midbrain reticular
  nucleus +0.25 and anterior pretectal nucleus +0.20, not in CA1 or the dentate gyrus; choice in all
  four (APN +0.29, MRN +0.30, CA1 +0.08, DG +0.11; the window includes the movement). Communication
  subspaces (`rrr.csv`): MRN and APN predict each other's held-out units with R² 0.10-0.13 through
  1-4 dimensions (within region: 0.36 for MRN, 0.17 for APN), and CA1 units at a similar level
  (0.08-0.13), while the dentate gyrus (16 units) is barely predicted by the others (R² ≤ 0.02).
- **Synthetic fixture** (tests): the planted structure is recovered: position R² > 0.5 (Bayes >
  0.6, CEBRA-kNN > 0.5) and held-out-session decoding beats no alignment; VISp decodes the stimulus
  side, MOs the choice and wheel, CA1 nothing (p > 0.05); the VISp-MOs shared latent gives a
  rank ≤ 2 communication subspace with R² above CA1's.

## Limitations

- Decoding shows that information is *present* in the recorded activity, linearly (ridge,
  logistic) or not (MLP, Bayes); it does not show that the brain uses it, and it can be carried
  by movement-related activity (choice decoded after the movement started). Choose
  `TRIAL_WINDOW_S` to ask a specific question (e.g. before movement onset).
- Behaviour variables correlate with each other (speed and position on a track, choice and wheel
  velocity); a decodable variable may be decoded through another one.
- The circular-shift null preserves the autocorrelation of both signals but not a periodic
  behaviour: on a very regular task a shift of one period still aligns them, so the null is
  conservative there. Trial-label shuffles ignore slow drifts (block structure).
- R² penalises bias: a decoder that tracks a variable well but is offset on one block can have a
  low or negative R² (see `r` and `median_abs_error`). Region scores depend on how many neurons a
  region has; compare with the neuron-dropping curve.
- The pseudo-session generator is a simplification of IBL's: choices come from the session's
  accuracy per contrast, not from a behavioural model fitted to the mouse (action kernel), and
  variables outside the task model (e.g. reward volume, contrast) use label permutations.
- Communication-subspace R² between regions is small for spontaneous activity and depends on the
  bin size and on shared slow drifts (behavioural state, movement); a low-rank between-region
  prediction is a statistical description, not proof of a direct connection.
- Held-out-session decoding uses labels of the held-out session's first half for the alignment
  (not for the decoder), and needs the variable in the same units in every session.

## Licences

Apache-2.0 (our code). Bundled: scikit-learn, NumPy, SciPy, pandas, pyarrow, Matplotlib (BSD /
Apache / PSF-style; see `THIRD_PARTY.md`). No GPL, no model weights.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in `CITATION.cff`.

- Glaser JI, Benjamin AS, Chowdhury RH, Perich MG, Miller LE, Kording KP. Machine Learning for
  Neural Decoding. *eNeuro* 7(4):ENEURO.0506-19.2020 (2020). doi:[10.1523/ENEURO.0506-19.2020](https://doi.org/10.1523/ENEURO.0506-19.2020), PMID: 32737181.
- Zhang K, Ginzburg I, McNaughton BL, Sejnowski TJ. Interpreting Neuronal Population Activity by
  Reconstruction: Unified Framework With Application to Hippocampal Place Cells. *Journal of
  Neurophysiology* 79:1017–1044 (1998). doi:[10.1152/jn.1998.79.2.1017](https://doi.org/10.1152/jn.1998.79.2.1017), PMID: 9463459.
- Gallego JA, Perich MG, Chowdhury RH, Solla SA, Miller LE. Long-term stability of cortical
  population dynamics underlying consistent behavior. *Nature Neuroscience* 23:260–270 (2020).
  doi:[10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4), PMID: 31907438.
- Schneider S, Lee JH, Mathis MW. Learnable latent embeddings for joint behavioural and neural
  analysis. *Nature* 617:360–368 (2023). doi:[10.1038/s41586-023-06031-6](https://doi.org/10.1038/s41586-023-06031-6), PMID: 37138088.
- Findling C, Hubert F, International Brain Laboratory, et al. Brain-wide representations of prior
  information in mouse decision-making. *Nature* 645:192–200 (2025). doi:[10.1038/s41586-025-09226-1](https://doi.org/10.1038/s41586-025-09226-1), PMID: 40903597. (pseudo-session nulls)
- Andre A, Mahato SP, Arora V, et al. BrainWideBench: Benchmarking large-scale pretraining and
  across-animal transfer in multi-region neural recordings. arXiv (2026). doi:[10.48550/arXiv.2609.22064](https://doi.org/10.48550/arXiv.2609.22064).
- Semedo JD, Zandvakili A, Machens CK, Yu BM, Kohn A. Cortical Areas Interact through a
  Communication Subspace. *Neuron* 102:249–259.e4 (2019). doi:[10.1016/j.neuron.2019.01.026](https://doi.org/10.1016/j.neuron.2019.01.026), PMID: 30770252.
- MacDowell CJ, Libby A, Jahn CI, Tafazoli S, Ardalan A, Buschman TJ. Multiplexed subspaces route
  neural activity across brain-wide networks. *Nature Communications* 16:3359 (2025). doi:[10.1038/s41467-025-58698-2](https://doi.org/10.1038/s41467-025-58698-2), PMID: 40204762.
- scikit-learn (Pedregosa et al., *JMLR* 12:2825–2830, 2011; no DOI): https://scikit-learn.org.
- Demo data: Grosmark AD, Buzsáki G. Diversity in neural firing dynamics supports both rigid and
  learned hippocampal sequences. *Science* 351:1440–1443 (2016). doi:[10.1126/science.aad1935](https://doi.org/10.1126/science.aad1935), PMID: 27013730;
  dataset DANDI:000044, doi:[10.48324/dandi.000044/0.261106.0236](https://doi.org/10.48324/dandi.000044/0.261106.0236).
- Demo data: International Brain Laboratory et al. A brain-wide map of neural activity during
  complex behaviour. *Nature* 645:177–191 (2025) (also the source of the corrected-score and
  Fisher-combination statistics). doi:[10.1038/s41586-025-09235-0](https://doi.org/10.1038/s41586-025-09235-0), PMID: 40903598;
  dataset DANDI:000409 (IBL - Brain Wide Map), doi:[10.48324/dandi.000409/0.260309.1324](https://doi.org/10.48324/dandi.000409/0.260309.1324).

## Acknowledgements

scikit-learn, NumPy, SciPy, pandas and Matplotlib are developed by their open-source communities;
the Bayesian decoder follows pynapple (Viejo, Peyrache and colleagues). The demo data come from the
Buzsáki lab (NYU; CRCNS hc-11 / DANDI:000044) and the International Brain Laboratory (DANDI:000409),
served by the DANDI Archive in the Neurodata Without Borders (NWB) format.
