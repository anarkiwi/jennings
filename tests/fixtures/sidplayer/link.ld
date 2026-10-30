__rc0 = 0xe0;
INCLUDE imag-regs.ld

MEMORY {
    ram (rwx) : ORIGIN = 0x1000, LENGTH = 0x1000
}

SECTIONS {
    .text : { KEEP(*(.text.entry)) *(.text .text.*) } >ram
    .rodata : { *(.rodata .rodata.*) } >ram
    .data : { *(.data .data.*) } >ram
    .bss (NOLOAD) : { *(.bss .bss.*) *(COMMON) } >ram
}

OUTPUT_FORMAT {
    SHORT(ORIGIN(ram))
    TRIM(ram)
}
