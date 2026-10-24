"""The search's candidate filter as a CUDA kernel (CuPy): one thread per candidate index,
the same pruning and semantics as :meth:`~.search.Search.candidates`."""

# pylint: disable=no-member

import functools

import cupy as cp
import numpy as np

from jennings.superopt.alphabet import MNEMONICS

MAX_LENGTH = 12
LAUNCH = 1 << 30
FOUND = 1 << 20
THREADS = 256

_SOURCE = r"""
#define NZ(x) { N = ((x) >> 7) & 1; Z = ((x) == 0); }
#define ADC(x) { int t_ = A + (x) + C; int r_ = t_ & 0xFF; \
    V = ((~(A ^ (x)) & (A ^ r_) & 0x80) != 0); C = t_ > 0xFF; A = r_; NZ(A); }
#define CMP(reg, x) { C = (reg) >= (x); NZ(((reg) - (x)) & 0xFF); }

extern "C" __global__ void filter(
    const int n, const long long start, const long long count, const int K,
    const int* codes, const int* sizes, const unsigned char* allowed,
    const unsigned char* last, const int budget,
    const int NV, const int S, const int I, const int T, const int R,
    const int* regs0, const int* slots0, const int* imms, const int* tables,
    const int O, const int* outidx, const int* expect,
    long long* found, unsigned int* nfound, const unsigned int maxfound)
{
    long long id = start + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= start + count) return;
    int d[MAXLEN];
    long long rest = id;
    for (int p = n - 1; p >= 0; p--) { d[p] = (int)(rest % K); rest /= K; }
    if (!last[d[n - 1]]) return;
    int total = 0;
    for (int p = 0; p < n; p++) {
        total += sizes[d[p]];
        if (p + 1 < n && !allowed[d[p] * K + d[p + 1]]) return;
    }
    if (total > budget) return;
    int m[MAXSLOTS];
    for (int v = 0; v < NV; v++) {
        int A = regs0[v * 7 + 0], X = regs0[v * 7 + 1], Y = regs0[v * 7 + 2];
        int C = regs0[v * 7 + 3], Z = regs0[v * 7 + 4], N = regs0[v * 7 + 5];
        int V = regs0[v * 7 + 6];
        for (int s = 0; s < S; s++) m[s] = slots0[v * S + s];
        for (int p = 0; p < n; p++) {
            const int* c = codes + 4 * d[p];
            int op = c[0], kind = c[1], a = c[2], b = c[3], val = 0;
            if (kind == 1) val = imms[v * I + a];
            else if (kind == 2) val = m[a];
            else if (kind == 3) val = tables[(v * T + a) * R + b + X];
            else if (kind == 4) val = tables[(v * T + a) * R + b + Y];
            int t, cin;
            switch (op) {
            case OP_LDA: A = val; NZ(A); break;
            case OP_LDX: X = val; NZ(X); break;
            case OP_LDY: Y = val; NZ(Y); break;
            case OP_LAX: A = X = val; NZ(A); break;
            case OP_STA: m[a] = A; break;
            case OP_STX: m[a] = X; break;
            case OP_STY: m[a] = Y; break;
            case OP_SAX: m[a] = A & X; break;
            case OP_TAX: X = A; NZ(X); break;
            case OP_TAY: Y = A; NZ(Y); break;
            case OP_TXA: A = X; NZ(A); break;
            case OP_TYA: A = Y; NZ(A); break;
            case OP_ADC: ADC(val); break;
            case OP_SBC: ADC(val ^ 0xFF); break;
            case OP_AND: A &= val; NZ(A); break;
            case OP_ORA: A |= val; NZ(A); break;
            case OP_EOR: A ^= val; NZ(A); break;
            case OP_CMP: CMP(A, val); break;
            case OP_CPX: CMP(X, val); break;
            case OP_CPY: CMP(Y, val); break;
            case OP_ASL: case OP_LSR: case OP_ROL: case OP_ROR:
            case OP_SLO: case OP_RLA: case OP_SRE: case OP_RRA:
                t = kind == 0 ? A : m[a]; cin = C;
                if (op == OP_ASL || op == OP_ROL || op == OP_SLO || op == OP_RLA) {
                    C = (t >> 7) & 1;
                    t = ((t << 1) | ((op == OP_ROL || op == OP_RLA) ? cin : 0)) & 0xFF;
                } else {
                    C = t & 1;
                    t = (t >> 1) | ((op == OP_ROR || op == OP_RRA) ? (cin << 7) : 0);
                }
                if (kind == 0) A = t; else m[a] = t;
                NZ(t);
                if (op == OP_SLO) { A |= t; NZ(A); }
                if (op == OP_RLA) { A &= t; NZ(A); }
                if (op == OP_SRE) { A ^= t; NZ(A); }
                if (op == OP_RRA) ADC(t);
                break;
            case OP_INC: m[a] = (m[a] + 1) & 0xFF; NZ(m[a]); break;
            case OP_DEC: m[a] = (m[a] + 255) & 0xFF; NZ(m[a]); break;
            case OP_DCP: m[a] = (m[a] + 255) & 0xFF; CMP(A, m[a]); break;
            case OP_ISC: m[a] = (m[a] + 1) & 0xFF; ADC(m[a] ^ 0xFF); break;
            case OP_INX: X = (X + 1) & 0xFF; NZ(X); break;
            case OP_INY: Y = (Y + 1) & 0xFF; NZ(Y); break;
            case OP_DEX: X = (X + 255) & 0xFF; NZ(X); break;
            case OP_DEY: Y = (Y + 255) & 0xFF; NZ(Y); break;
            case OP_CLC: C = 0; break;
            case OP_SEC: C = 1; break;
            case OP_ANC: A &= val; NZ(A); C = N; break;
            case OP_ALR: t = A & val; C = t & 1; A = t >> 1; NZ(A); break;
            case OP_SBX: t = A & X; CMP(t, val); X = (t - val) & 0xFF; break;
            }
        }
        int state[7] = {A, X, Y, C, Z, N, V};
        for (int o = 0; o < O; o++) {
            int k = outidx[o];
            int got = k < 7 ? state[k] : m[k - 7];
            if (got != expect[v * O + o]) return;
        }
    }
    unsigned int slot = atomicAdd(nfound, 1u);
    if (slot < maxfound) found[slot] = id;
}
"""


@functools.lru_cache(maxsize=4)
def _kernel(slots: int):
    defines = "".join(f"#define OP_{m} {k}\n" for k, m in enumerate(MNEMONICS))
    defines += f"#define MAXLEN {MAX_LENGTH}\n#define MAXSLOTS {max(slots, 1)}\n"
    return cp.RawKernel(defines + _SOURCE, "filter")


def candidates(search, n: int, budget: int):
    """``(survivors, tried)`` as :meth:`~.search.Search.candidates`, on the GPU."""
    # pylint: disable=too-many-locals
    if n > MAX_LENGTH:
        raise ValueError(f"length {n} past {MAX_LENGTH}")
    alphabet, vec = search.alphabet, search.vectors
    k = len(alphabet)
    kernel = _kernel(len(alphabet.slots))
    put = functools.partial(cp.asarray)
    args = (
        put(alphabet.codes.ravel()),
        put(alphabet.sizes),
        put(search.allowed.ravel()),
        put(search.last),
        np.int32(budget),
        np.int32(vec.count),
        np.int32(vec.slots.shape[1]),
        np.int32(vec.imms.shape[1]),
        np.int32(vec.tables.shape[1]),
        np.int32(vec.tables.shape[2]),
        put(vec.regs.ravel()),
        put(vec.slots.ravel()),
        put(vec.imms.ravel()),
        put(vec.tables.ravel()),
        np.int32(len(search.out_index)),
        put(np.array(search.out_index, dtype=np.int32)),
        put(vec.expect.ravel()),
    )
    found = cp.zeros(FOUND, dtype=cp.int64)
    survivors, total = [], k**n
    for start in range(0, total, LAUNCH):
        count = min(LAUNCH, total - start)
        nfound = cp.zeros(1, dtype=cp.uint32)
        blocks = (count + THREADS - 1) // THREADS
        kernel(
            (blocks,),
            (THREADS,),
            (np.int32(n), np.int64(start), np.int64(count), np.int32(k))
            + args
            + (found, nfound, np.uint32(FOUND)),
        )
        hits = int(nfound.get()[0])
        if hits > FOUND:
            raise RuntimeError(f"{hits} survivors overflow the buffer")
        for index in found[:hits].get().tolist():
            survivors.append(tuple((index // k**p) % k for p in range(n - 1, -1, -1)))
    return sorted(survivors), total
