"""ACME sources with the files acme 0.97 writes for them, and sources it rejects."""

SOURCES = {
    "player": """!to "player.sid", plain
!cpu 6510
ADDR = $1000
* = 0
!text "PSID"
!be16 2, $7c
!be16 init, play
- !text "name"
!fill 8-(*--), 0
!wo ADDR
!pseudopc ADDR {
TRACK = $f0
init    jmp _init
play    inc frc
        lda (TRACK),y
        bne +
        jmp ++
+       sta tab,x
++      ldx #2
-       dec tab,x
        bpl -
        isc frc+1
        dcp tab,x
_init   lda #%01000000
        rts
!macro INSTR .i, .l {
    !by $80 | .l, .i
}
tab     !h 00 01 02
frc     !h 0003
        +INSTR 2, 5
        +INSTR 3, 1
} ; end
""",
    "force": """*=$1000
a = $0012
b+2 = 3
c = <$1234
d = $000012
lda+2 $12
lda a
lda b
sta c,x
lda+1 c
jmp ($1234)
e+1 = 5
!be24 $123456 : !le24 1 : !32 -1 : !be32 $12345678 : !16 $ffff, -1, 'a'
!by -128, 255, "x", 'y'
!tx "a\\"b\\\\c\\n", 0, "\\0\\t\\r'"
""",
    "local": """*=$2000
!macro twice .v {
.l  lda #.v
    bne .l
}
!macro g x, .y {
    !by x, .y
}
.top nop
    +twice 1
    +twice 2
    +g 3, 4
    lda .top
    ldx x
glob:
.top2 = glob + 1
    jmp .top2
""",
    "pseudo": """* = $1000
outer = *
!pseudopc $8000 {
in1 nop
!pseudopc $9000 { in2 jmp in2 }
* = $8008
in3 rts
}
after !word in1, in2, in3, after
* = $1010
!cpu 6502 { lda #0 }
!cpu 6510
lax $12 : asr #$f0 : anc #1 : sbx #2 : sha $1234,y : jam : nop #1 : nop $12,x
""",
    "exprs": """*=$1000
!by 1+2*3, (1+2)*3, 7/2, -7/2, 7%3, -7%3, 2^3, 1<<3, 256>>4, $80>>>1
!by 3==3, 3=4, 3!=4, 3<>4, 3><3, 2<3, 3<=3, 4>3, 3>=4, not 0 & 1, !0 & 255
!by 5 xor 3, 5 or 2, 6 and 3, 7 div 2, 7 mod 4, 1 asl 2, 8 lsr 1, 8 asr 1, 1 lsl 1, 5 eor 1
!wo ^$123456, >$1234, <$1234, -1 & $ffff, --5, ---5 & $ff, 2^3^2, -2^2 & $ff
!by %1010, %..##..##, &17, 0x1f, 0b101, $0a, 1 << 65, -8 >> 1 & $ff
""",
    "anon": """*=$1000
-   dex
    bne -
-- inx
    bne --
    beq +
    beq ++
    nop
+   nop
++  nop
+   bne +
+   rts
    !wo -, --, +
+
""",
    "zpfwd": """*=$1000
  lda fwd
  lda fwd,x
  lda <fwd
  sta (fwd),y
  lda fwd2
fwd = $12
fwd2 = $1234
  lda fwd
  !fill 3
  !fill 2, $ea
""",
    "crlf": "#!/usr/bin/acme\r\n*=$1000\r\n  lda $12 // c\r  !8 1,2 : !08 3\r\n  !by 6 / 2\n",
    "segs": """*=$1010
!by 1
*=$1000
!by 2
*=$1005, overlay
!by 3, <*
""",
    "cbm": """!to "a.prg"
*=$c000
lda #1 : asl : rol : ldx $12,y : stx $1234 : lda ($12,x)
""",
    "apple": """\t!to "a.bin", apple ; x
*=$0801 : !wo 1, 2 :rts
""",
    "empty": "x = 1\n",
    "trees": "smk = 1\nsam = 2\nsoj = 3\nsgn = 4\nscl = 5\nskh = 6\n"
    "sii = 7\nseo = 8\n*=0\n!by sam, soj\n",
    "macros": """*=$80
!macro .loc .v {
    !tx "}{", .v
    !cpu 6510 { lax $12 : lda #"a" : lda "b" }
}
start+2 nop
    +.loc 7
    lda start
start2+2 = $12
    lda start2
    lda fwd
fwd = $34
""",
}


EXPECTED = {
    "player": {
        "labels": (
            "\tADDR\t= $1000\n"
            "\tinit\t= $1000\t; ?\n"
            "\t_init\t= $101d\t; ?\n"
            "\tTRACK\t= $f0\n"
            "\tfrc\t= $1023\t; ?\n"
            "\ttab\t= $1020\t; ?\n"
            "\tplay\t= $1003\t; ?\n"
        ),
        "out": bytes.fromhex(
            "505349440002007c100010036e616d650000000000104c1d10ee2310b1f0d003"
            "4c10109d2010a202de201010fbef2410df2010a94060000102000385028103"
        ),
        "player.sid": bytes.fromhex(
            "505349440002007c100010036e616d650000000000104c1d10ee2310b1f0d003"
            "4c10109d2010a202de201010fbef2410df2010a94060000102000385028103"
        ),
    },
    "force": {
        "labels": (
            "\ta+2\t= $12\n"
            "\tb+2\t= $3\n"
            "\tc\t= $34\n"
            "\td+3\t= $12\t; unused\n"
            "\te\t= $5\t; unused\n"
        ),
        "out": bytes.fromhex(
            "ad1200ad1200ad03009534a5346c3412123456010000ffffffff12345678ffff"
            "ffff610080ff78796122625c630a0000090d27"
        ),
    },
    "local": {
        "labels": ("\tglob\t= $2010\n" "\tx\t= $3\n"),
        "out": bytes.fromhex("eaa901d0fca902d0fc0304ad0020a6034c1120"),
    },
    "pseudo": {
        "labels": (
            "\tafter\t= $1009\n"
            "\tin1\t= $8000\n"
            "\tin3\t= $8008\n"
            "\tin2\t= $9000\n"
            "\touter\t= $1000\t; unused\n"
        ),
        "out": bytes.fromhex(
            "ea4c0090000000006000800090088009a900a7124bf00b01cb029f34120280011412"
        ),
    },
    "exprs": {
        "labels": "",
        "out": bytes.fromhex(
            "070903fd01ff0808104001000101000101010001ff0607020303040404020412"
            "0012003400ffff0500fb000002fc000a330f1f050a02fc"
        ),
    },
    "anon": {
        "labels": "",
        "out": bytes.fromhex("cad0fde8d0fdf003f002eaeaead00060001003101610"),
    },
    "zpfwd": {
        "labels": ("\tfwd2\t= $1234\t; ?\n" "\tfwd\t= $12\t; ?\n"),
        "out": bytes.fromhex("ad1200bd1200a5129112ad3412ad1200000000eaea"),
    },
    "crlf": {
        "labels": "",
        "out": bytes.fromhex("a51201020303"),
    },
    "segs": {
        "labels": "",
        "out": bytes.fromhex("0200000000030500000000000000000001"),
    },
    "cbm": {
        "a.prg": bytes.fromhex("00c0a9010a2ab6128e3412a112"),
        "labels": "",
        "out": bytes.fromhex("a9010a2ab6128e3412a112"),
    },
    "apple": {
        "a.bin": bytes.fromhex("010805000100020060"),
        "labels": "",
        "out": bytes.fromhex("0100020060"),
    },
    "empty": {
        "labels": ("\tx\t= $1\t; unused\n"),
        "out": bytes.fromhex(""),
    },
    "trees": {
        "labels": (
            "\tsmk\t= $1\t; unused\n"
            "\tsoj\t= $3\n"
            "\tsam\t= $2\n"
            "\tsgn\t= $4\t; unused\n"
            "\tskh\t= $6\t; unused\n"
            "\tsii\t= $7\t; unused\n"
            "\tscl\t= $5\t; unused\n"
            "\tseo\t= $8\t; unused\n"
        ),
        "out": bytes.fromhex("0203"),
    },
    "macros": {
        "labels": ("\tfwd\t= $34\t; ?\n" "\tstart2+2\t= $12\n" "\tstart+2\t= $80\n"),
        "out": bytes.fromhex("ea7d7b07a712a961a562ad8000ad1200ad3400"),
    },
}

ERRORS = {
    "*=$1000\nlda nowhere\n": "2: Value not defined (nowhere).",
    "*=$1000\nlda #$00ff\n": "Number out of range.",
    "*=$1000\nlda ($0012),y\n": "Number out of range.",
    "*=$1000\nldx $000012,y\n": "Number out of range.",
    "*=$1000\nbne $2000\n": "Target out of range",
    "*=$1000\nbne $12345\n": "Target not in bank",
    "a = 1\na = 2\n": "Symbol already defined.",
    "nop\n": "Program counter undefined.",
    "!macro a {\n}\n!macro a {\n}\n": "Macro already defined.",
    "*=$1000\n+a 1\n": "Macro not defined (or wrong signature).",
    "*=$1000\nlda -1\n": "Negative value - cannot choose addressing mode.",
    "*=$fffe\n!by 1,2,3\n": "Produced too much code.",
    "*=$1000\nlax $12\n": "Syntax error.",
    "*=$1000\n!by 256\n": "Number out of range (8 bits).",
    "*=$1000\n!wo $10000\n": "Number out of range (16 bits).",
    '*=$1000\n!by "ab"\n': "There's more than one character.",
    "*=$1000\n!by 1/0\n": "Division by zero.",
    "*=$1000\n!by 2^-1\n": "Exponent is negative.",
    "*=$1000\nlda (1,x\n": "Syntax error.",
    "*=$1000\nlda ((1,x)\n": "Too many '('.",
    "*=$1000\n!by (1\n": "Too many '('.",
    "*=$1000\nlda $12 $13\n": "Garbage data at end of statement.",
    "*=$1000\nlda #1,x\n": "Garbage data at end of statement.",
    "*=$1000\nsta #1\n": "CPU does not support this addressing mode for this mnemonic.",
    "*=$1000\nlda ($12,y)\n": "CPU does not support this addressing mode for this mnemonic.",
    "*=$1000\nlda $12,z\n": "Syntax error.",
    "*=$1000\njmp+1 $12\n": "CPU does not support this postfix for this mnemonic.",
    "*=$1000\nlda+4 $12\n": "Illegal postfix.",
    "a\na = 1\n": "Symbol already defined.",
    "*=$1000\na = 1\n!by a\na+2 = 1\n": "Too late for postfix.",
    "x a\n": "Syntax error.",
    "*=$1000\n!h 1\n": "Hex digits are not given in pairs.",
    "*=$1000\n!h 12,\n": "Syntax error.",
    "*=$1000\n!fill -1\n": "Negative size argument.",
    "*=$1000\n!fill fwd\nfwd = 1\n": "Value not defined (fwd).",
    "!to x\n": "File name quotes not found.",
    '!to ""\n': "No file name given.",
    '!to "x", d64\n': "Unknown output format.",
    '*=$1000\n!by "a\n': "Quotes still open at end of line.",
    "*=$1000\n!by 'ab'\n": "There's more than one character.",
    "*=$1000\n!by ''\n": "No string given.",
    '*=$1000\n!tx "\\q"\n': "Unsupported backslash sequence.",
    "*=$1000\n!by $\n": "Literal without any digits.",
    "*=$1000\n!by 1 foo 2\n": "Unknown operator.",
    "*=$1000\n!by 1 ! 2\n": "Syntax error.",
    "*=$1000\n!by ,\n": "No value given.",
    "*=$1000\n!by #\n": "Syntax error.",
    '*=$1000\n!by "a"+1\n': "Unsupported: string operations.",
    '*=$1000\n!by -"a"\n': "Unsupported: string operations.",
    '*=$1000\n!by "ab" : lda #"ab"\n': "There's more than one character.",
    '*=$1000\nlda #"ab"\n': "String length is not 1.",
    "* $1000\n": "Syntax error.",
    "*=$1000, far\n": 'Unknown "*=" segment modifier.',
    "*=$1000\n}\n": "Found '}' instead of end-of-file.",
    "!cpu 6510 {\n": "Found end-of-file instead of '}'.",
    "!macro m {\n": "Found end-of-file instead of '}'.",
    "!macro m .a\n": "No left brace found.",
    "!macro m {\n+m\n}\n*=$1000\n+m\n": "Too deeply nested. Recursive macro calls?",
    "!cpu z80\n": "Unknown processor.",
    "!foo\n": "Unknown pseudo opcode.",
    "!\n": "No string given.",
}
UNSUPPORTED = {
    "!zone a\n": "!zone",
    "!for i, 1, 2 {\n}\n": "!for",
    '!binary "x"\n': "!binary",
    "!cpu 65c02\n": "!cpu 65c02",
    "@a = 1\n": "cheap locals",
    "a = @b\n": "cheap locals",
    "a = [1]\n": "lists",
    "a = 1.5\n": "floating point",
    "a = .5\n": "floating point",
    "a = sin(1)\n": "function sin()",
    "a = &b\n": "&",
    'a = "s"\n': "string symbols",
    "!macro m ~.a {\n}\n": "macro parameter ~",
    "!macro m @a {\n}\n": "macro parameter @",
    "!macro m .a {\n}\n+m ~b\n": "call by reference",
    '!macro m .a {\n}\n+m "ab"\n': "string macro arguments",
    "*=$1000\n!pseudopc $2000\n": "!pseudopc without a block",
    "!cpu 6510\n*=$1000\ndop\n": "mnemonic dop",
    "*=$1000\nlda [$12]\n": "long indirect addressing",
    "a = b[1]\n": "indexing",
}
