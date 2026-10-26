FROM nvidia/cuda:13.4.3-devel-ubuntu24.04 AS base
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir "cupy-cuda12x>=13" "z3-solver>=4.12" numpy pytest pytest-xdist \
    "deity-informant>=0.4.1" py65

FROM base
WORKDIR /src
COPY . /src
RUN pip install --no-cache-dir --no-deps -e .
CMD ["pytest", "-q", "tests/test_superopt.py"]
