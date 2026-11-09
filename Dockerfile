FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# The complete pinned dependency set, no resolver (see requirements.txt). No PyTorch: every decoder here is
# scikit-learn / numpy.
COPY requirements.txt /processor/
RUN pip install --no-deps --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

# Build matplotlib's font cache and import the stack once, so a run needs no writes outside /tmp and no network.
RUN python -c "import matplotlib.pyplot, sklearn.neural_network, sklearn.linear_model, pyarrow, processor.main" \
    && mkdir -p /tmp/matplotlib /tmp/cache && chmod -R a+rwX /tmp/matplotlib /tmp/cache

RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
