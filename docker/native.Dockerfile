ARG PYTHON_VERSION=3.12

FROM rust:1.99-slim-trixie AS rust

FROM python:${PYTHON_VERSION}-slim-trixie AS pydeps
WORKDIR /src
COPY pyproject.toml .
# deity-informant depends on jennings, so pip pulls a released jennings that is then removed.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /requirements.txt \
    && pip install --no-cache-dir -r /requirements.txt \
    && pip uninstall -y jennings

FROM python:${PYTHON_VERSION}-slim-trixie AS toolchain
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=rust /usr/local/cargo /usr/local/cargo
RUN pip install --no-cache-dir maturin

FROM toolchain AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/jennings-core/Cargo.toml crates/jennings-core/
COPY crates/jennings-py/Cargo.toml crates/jennings-py/
RUN mkdir -p crates/jennings-core/src crates/jennings-py/src \
    && touch crates/jennings-core/src/lib.rs crates/jennings-py/src/lib.rs \
    && cargo fetch --locked
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY crates crates
COPY jennings jennings
RUN maturin build --release --locked -i python -o /wheels

FROM pydeps AS test
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl \
    && python -c "import jennings.memory as m; assert m.NATIVE"
# Only tests and tools are copied so imports resolve to the installed wheel.
COPY tests tests
COPY tools tools
COPY crates/jennings-core/src/cpu_gen.rs crates/jennings-core/src/cpu_gen.rs
CMD ["sh", "-c", "pytest tests -n auto -q && python tools/gen_cpu.py --check"]
