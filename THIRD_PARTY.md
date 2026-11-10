# Third-party software

This app is Apache-2.0 (see `LICENSE`). Its image contains the following third-party components,
installed from PyPI at the pinned versions in `requirements.txt`; their licences apply to the image.
Nothing here is GPL, and there is no PyTorch and no model weight: every decoder is fitted on the
user's data at run time. The decoders are written here on top of scikit-learn and NumPy; the
Bayesian position decoder follows Zhang et al. (1998) and pynapple's `decode_1d` (no pynapple code
is copied or bundled), held-out-session alignment follows Gallego et al. (2020).

| Component | What it does here | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| scikit-learn 1.9.1 | Ridge (`RidgeCV`), MLP (`MLPRegressor`), logistic regression, kNN, PCA, balanced accuracy. | The scikit-learn developers | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, linear algebra, smoothing. | NumPy and SciPy developers | BSD-3-Clause (NumPy also bundles 0BSD/MIT/Zlib/CC0 parts) | https://numpy.org, https://scipy.org |
| pandas 3.1.1, pyarrow 26.0.1 | events.parquet (trials), tables. | pandas developers; Apache Software Foundation | BSD-3-Clause; Apache-2.0 | https://pandas.pydata.org, https://arrow.apache.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | PNG figures. | Matplotlib Development Team; Jeffrey A. Clark and contributors | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org |
| joblib, threadpoolctl, cloudpickle, narwhals, contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, python-dateutil, tzdata | Dependencies of the above (versions in `requirements.txt`). | Their authors | BSD / MIT / Apache-2.0 / PSF-2.0 | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12). | Python Software Foundation; Debian | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

When neuro-embedding's output is linked, this app reads CEBRA embeddings from files; CEBRA itself
(Apache-2.0, EPFL patent WO2023143843A1, see neuro-embedding's THIRD_PARTY.md) is not in this image.

## Not bundled

- **Demo data** (not in the image or the repository): DANDI:000044 (Grosmark & Buzsáki 2016,
  CC-BY-4.0, doi:10.48324/dandi.000044/0.261106.0236) and DANDI:000409 (International Brain
  Laboratory Brain Wide Map, CC-BY-4.0, doi:10.48324/dandi.000409/0.260309.1324), read through
  neuro-align.
