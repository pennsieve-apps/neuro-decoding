# sample-input

This app reads the `aligned` port: the output of **neuro-align** (contract in
`workflow-systems-neuroscience/docs/conventions.md`, "Decoding"), one folder per session:

    neuro-align/<session_id>/aligned.npz     t, X (bins x neurons), Y (bins x behaviour variables), neuron_ids,
                                             neuron_meta, behavior_names/units/kinds, epoch_mask, bin_size
    neuro-align/<session_id>/events.parquet  trials and other TimeIntervals (trial variables are decoded per trial)
    neuro-align/<session_id>/spikes.npz      spike times (times + offsets)
    neuro-align/<session_id>/manifest.json

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/neuro-decoding/
```

Three sessions in neuro-align 0.1.0's exact format: two rats on a 160 cm linear track
(`sub-rat1_ses-track1`, `sub-rat2_ses-track1`; direction-selective place cells plus unrelated
units, irregular laps), and an IBL-like task session (`sub-mouse1_ses-ibl1`; VISp units respond
to the stimulus side, MOs units to the choice and wheel velocity, CA1 units to nothing; a trials
table). The fixture with `--embedding` also has a planted neuro-embedding output (a noisy 3-D ring of position) for the optional `embedding` port. `manifest.json` → `synthetic_truth` holds what was planted.

## Real data

Run neuro-align on any NWB file with units (or ophys ROIs) and behaviour, for example:

- **Grosmark & Buzsáki 2016, DANDI:000044** (CC-BY-4.0): `sub-Buddy/sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb`
  (5.4 GB; CA1 units on a 1.6 m linear track; the rats of CEBRA's "rat hippocampus" benchmark),
  with neuro-align `EPOCH=run`. In the workflow repository: `data/grosmark/`.
- **IBL Brain Wide Map, DANDI:000409** (CC-BY-4.0): any `..._desc-processed_behavior+ecephys.nwb`
  (units with brain regions, trials, wheel, pose); `data/ibl/` holds
  `sub-NR-0031_ses-caa5dddc-..._desc-processed_behavior+ecephys.nwb` (512 MB).

Download with the DANDI CLI (`dandi download DANDI:000044/0.261106.0236 --existing skip`, or the asset URL from
https://dandiarchive.org/dandiset/000044).
