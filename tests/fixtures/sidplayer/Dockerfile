FROM ghcr.io/anarkiwi/docker-mos-llvm-sdk:v23.0.1 AS toolchain
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

FROM toolchain AS build
WORKDIR /work
COPY Makefile link.ld player.s freq.c ./
RUN make

FROM scratch AS artifacts
COPY --from=build /work/sidplayer.prg /work/sidplayer.prg.elf /
