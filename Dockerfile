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
    MPLBACKEND=Agg \
    USER=app \
    LOGNAME=app

# Import the stack once at build time (fails the build on a broken install). Nothing is baked into /tmp or
# HOME: on Lambda only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR are writable and /tmp starts empty, so
# processor/entry.py points matplotlib, numba, torch and XDG caches below $TMPDIR at start-up. No model
# weights are needed (every model is fitted on the user's data).
RUN MPLCONFIGDIR=/dev/shm/mpl python -W ignore -c "import matplotlib.pyplot, sklearn, pyarrow, processor.main"

RUN useradd --uid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
