CC = mos-c64-clang
CFLAGS = -g -Os
LDFLAGS = -nostartfiles -nostdlib -T link.ld -Wl,-e,init

all: sidplayer.prg

sidplayer.prg: player.s freq.c link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ player.s freq.c

.PHONY: all
