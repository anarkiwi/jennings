; Three-voice SID playroutine: init (A = song), play (once per frame).
;
; Pattern bytes: $00 rest, $01-$53 note, $60-$7f duration ($60 = 1 tick),
; $80-$8f instrument, $90-$9f effect with one parameter byte, $ff end.
; Sequence bytes: pattern number, $ff restart.

SID = 0xd400
NVOICES = 3

.section .text.entry,"ax",@progbits
.globl init
init:
        jmp     start
.globl play
play:
        jmp     frame

.text
start:
        tax
        lda     song_speed,x
        sta     speed
        lda     #1
        sta     tick
        stx     tmp
        txa
        asl
        adc     tmp
        asl
        tay
        ldx     #0
init_voice:
        lda     song_seq,y
        sta     seq_lo,x
        lda     song_seq+1,y
        sta     seq_hi,x
        lda     #0
        sta     seq_pos,x
        sta     vib_depth,x
        sta     slide,x
        sta     ctrl,x
        lda     #1
        sta     dur,x
        sta     dur_left,x
        tya
        pha
        jsr     next_pattern
        pla
        tay
        iny
        iny
        inx
        cpx     #NVOICES
        bne     init_voice
        lda     #0x0f
        sta     SID+24
        rts

; X = voice: load the next pattern of the voice's sequence.
next_pattern:
        lda     seq_lo,x
        sta     seq_rd+1
        lda     seq_hi,x
        sta     seq_rd+2
        ldy     seq_pos,x
seq_rd:
        lda     0xffff,y
        iny
        cmp     #0xff
        bne     seq_have
        ldy     #0
        beq     seq_rd
seq_have:
        asl
        sta     tmp
        tya
        sta     seq_pos,x
        ldy     tmp
        lda     pat_tab,y
        sta     pat_lo,x
        lda     pat_tab+1,y
        sta     pat_hi,x
        lda     #0
        sta     pat_pos,x
        rts

; Read the pattern byte at Y of the patched pointer and advance.
getb:
pat_rd:
        lda     0xffff,y
        iny
        rts

; X = voice: consume pattern bytes up to and including the next note or rest.
fetch:
        lda     pat_lo,x
        sta     pat_rd+1
        lda     pat_hi,x
        sta     pat_rd+2
        ldy     pat_pos,x
next_byte:
        jsr     getb
        cmp     #0xff
        beq     end_pattern
        cmp     #0x90
        bcs     effect
        cmp     #0x80
        bcs     instrument
        cmp     #0x60
        bcs     duration
        sta     tmp
        tya
        sta     pat_pos,x
        lda     tmp
        beq     rest
        jmp     note_on
end_pattern:
        jsr     next_pattern
        jmp     fetch
duration:
        sbc     #0x5f
        sta     dur,x
        jmp     next_byte
instrument:
        and     #0x0f
        sta     inst,x
        jmp     next_byte
effect:
        and     #0x0f
        asl
        sta     tmp
        jsr     getb
        sty     ysave
        ldy     tmp
        jsr     dispatch
        ldy     ysave
        jmp     next_byte
dispatch:
        sta     param
        lda     fx_table+1,y
        pha
        lda     fx_table,y
        pha
        lda     param
        rts
rest:
        lda     ctrl,x
        and     #0xfe
        sta     ctrl,x
        lda     dur,x
        sta     dur_left,x
        rts
note_on:
        stx     xsave
        jsr     note_freq
        stx     tmp
        ldx     xsave
        sta     tgt_lo,x
        lda     tmp
        sta     tgt_hi,x
        lda     slide,x
        bne     glide
        lda     tgt_lo,x
        sta     frq_lo,x
        lda     tgt_hi,x
        sta     frq_hi,x
glide:
        lda     #0
        sta     vib_phase,x
        ldy     inst,x
        lda     inst_ad,y
        sta     ad,x
        lda     inst_sr,y
        sta     sr,x
        lda     inst_pulse,y
        sta     pulse,x
        lda     inst_wave,y
        ora     #1
        sta     ctrl,x
        lda     dur,x
        sta     dur_left,x
        rts

fx_off:
        lda     #0
        sta     vib_depth,x
        sta     slide,x
        rts
fx_vibrato:
        sta     vib_depth,x
        rts
fx_slide:
        sta     slide,x
        rts
fx_pulse:
        sta     pulse,x
        rts

frame:
        dec     tick
        bne     voices
        lda     speed
        sta     tick
        ldx     #NVOICES-1
tick_voice:
        dec     dur_left,x
        bne     tick_next
        jsr     fetch
tick_next:
        dex
        bpl     tick_voice
voices:
        ldx     #NVOICES-1
voice:
        lda     voice_reg,x
        sta     sidw+1
        lda     slide,x
        beq     vibrato
        lda     frq_lo,x
        cmp     tgt_lo,x
        lda     frq_hi,x
        sbc     tgt_hi,x
        bcs     slide_down
        clc
        lda     frq_lo,x
        adc     slide,x
        sta     frq_lo,x
        lda     frq_hi,x
        adc     #0
        sta     frq_hi,x
        lda     tgt_lo,x
        cmp     frq_lo,x
        lda     tgt_hi,x
        sbc     frq_hi,x
        bcs     vibrato
        jmp     snap
slide_down:
        sec
        lda     frq_lo,x
        sbc     slide,x
        sta     frq_lo,x
        lda     frq_hi,x
        sbc     #0
        sta     frq_hi,x
        bcc     snap
        lda     frq_lo,x
        cmp     tgt_lo,x
        lda     frq_hi,x
        sbc     tgt_hi,x
        bcs     vibrato
snap:
        lda     tgt_lo,x
        sta     frq_lo,x
        lda     tgt_hi,x
        sta     frq_hi,x
vibrato:
        lda     vib_depth,x
        beq     steady
        inc     vib_phase,x
        lda     vib_phase,x
        and     #4
        bne     vib_down
        clc
        lda     frq_lo,x
        adc     vib_depth,x
        sta     out_lo
        lda     frq_hi,x
        adc     #0
        sta     out_hi
        jmp     write
vib_down:
        sec
        lda     frq_lo,x
        sbc     vib_depth,x
        sta     out_lo
        lda     frq_hi,x
        sbc     #0
        sta     out_hi
        jmp     write
steady:
        lda     frq_lo,x
        sta     out_lo
        lda     frq_hi,x
        sta     out_hi
write:
        ldy     #0
        lda     out_lo
        jsr     sidw
        iny
        lda     out_hi
        jsr     sidw
        ldy     #3
        lda     pulse,x
        jsr     sidw
        iny
        lda     ctrl,x
        jsr     sidw
        iny
        lda     ad,x
        jsr     sidw
        iny
        lda     sr,x
        jsr     sidw
        dex
        bmi     done
        jmp     voice
done:
        rts

; A = value, Y = register; the operand low byte selects the voice.
sidw:
        sta     SID,y
        rts

.section .rodata,"a",@progbits
voice_reg:
        .byte   0, 7, 14
fx_table:
        .short  fx_off-1, fx_vibrato-1, fx_slide-1, fx_pulse-1

inst_ad:
        .byte   0x09, 0x28, 0x00, 0x02
inst_sr:
        .byte   0x00, 0xa9, 0xf4, 0x00
inst_wave:
        .byte   0x40, 0x10, 0x20, 0x80
inst_pulse:
        .byte   0x08, 0x00, 0x00, 0x00

song_speed:
        .byte   6, 4
song_seq:
        .short  seq0, seq1, seq2
        .short  seq3, seq4, seq5

seq0:   .byte   0, 0, 1, 0xff
seq1:   .byte   2, 3, 0xff
seq2:   .byte   4, 0xff
seq3:   .byte   1, 5, 0xff
seq4:   .byte   6, 0xff
seq5:   .byte   4, 7, 0xff

pat_tab:
        .short  pat0, pat1, pat2, pat3, pat4, pat5, pat6, pat7

pat0:   .byte   0x80, 0x61, 24, 36, 24, 36, 29, 41, 31, 43, 0xff
pat1:   .byte   0x80, 0x61, 33, 45, 33, 45, 29, 41, 31, 0, 0xff
pat2:   .byte   0x81, 0x63, 0x91, 0x30, 48, 52, 55, 0x67, 60, 0x90, 0, 0x63, 59, 55, 0xff
pat3:   .byte   0x81, 0x63, 0x91, 0x40, 57, 55, 53, 0x67, 52, 0x90, 0, 0x63, 0, 50, 0xff
pat4:   .byte   0x83, 0x61, 70, 0, 70, 70, 0xff
pat5:   .byte   0x80, 0x63, 36, 31, 33, 29, 0xff
pat6:   .byte   0x82, 0x63, 0x92, 0x40, 60, 64, 67, 0x67, 72, 0x90, 0, 0x63, 67
        .byte   0x92, 0x20, 60, 0xff
pat7:   .byte   0x83, 0x93, 0x04, 0x61, 70, 70, 0x93, 0x08, 70, 0, 0xff

.section .bss,"aw",@nobits
speed:          .zero   1
tick:           .zero   1
tmp:            .zero   1
xsave:          .zero   1
ysave:          .zero   1
param:          .zero   1
out_lo:         .zero   1
out_hi:         .zero   1
seq_lo:         .zero   NVOICES
seq_hi:         .zero   NVOICES
seq_pos:        .zero   NVOICES
pat_lo:         .zero   NVOICES
pat_hi:         .zero   NVOICES
pat_pos:        .zero   NVOICES
dur:            .zero   NVOICES
dur_left:       .zero   NVOICES
inst:           .zero   NVOICES
tgt_lo:         .zero   NVOICES
tgt_hi:         .zero   NVOICES
frq_lo:         .zero   NVOICES
frq_hi:         .zero   NVOICES
slide:          .zero   NVOICES
vib_depth:      .zero   NVOICES
vib_phase:      .zero   NVOICES
pulse:          .zero   NVOICES
ctrl:           .zero   NVOICES
ad:             .zero   NVOICES
sr:             .zero   NVOICES
