//! First repeats of projections of trajectory rows, and where each stops repeating with its
//! period, on the rows' columns.

use crate::taint::FxMap;

const NONE: u32 = u32::MAX;
const K: u64 = 0x9E37_79B9_7F4A_7C15;

/// A projection's first repeat and the first break of its period after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Period {
    /// `(preperiod, period)` of the first row equal on the columns to an earlier one.
    pub first: Option<(usize, usize)>,
    /// The first `t` from the preperiod on whose row equals row `t + period` on the columns
    /// while row `t + 1` differs from row `t + 1 + period`.
    pub conflict: Option<usize>,
}

/// The [`Period`] of each column set of `subsets` over the rows whose columns are `cols`,
/// all of one length.
pub fn periods(cols: &[&[u8]], subsets: &[Vec<usize>]) -> Vec<Period> {
    let n = cols.first().map_or(0, |c| c.len());
    let mut hash = Vec::new();
    subsets
        .iter()
        .map(|s| {
            let set: Vec<&[u8]> = s.iter().map(|&c| cols[c]).collect();
            period_of(&set, n, &mut hash)
        })
        .collect()
}

/// The [`Period`] of the columns `cols`, each one byte per row of `n`: rows are hashed a
/// block at a time, blocks doubling from 1024, until the first repeat, so the search costs
/// the repeat's position rather than `n`.
fn period_of(cols: &[&[u8]], n: usize, hash: &mut Vec<u64>) -> Period {
    let same = |a: usize, b: usize| cols.iter().all(|c| c[a] == c[b]);
    let (mut head, mut next) = (FxMap::<u64, u32>::default(), Vec::new());
    let (mut first, mut lo, mut block) = (None, 0, 1024);
    'blocks: while lo < n && first.is_none() {
        let hi = (lo + block).min(n);
        hash.clear();
        hash.resize(hi - lo, K);
        for col in cols {
            for (h, &v) in hash.iter_mut().zip(&col[lo..hi]) {
                *h = (*h ^ v as u64).wrapping_mul(K);
            }
        }
        for (i, &h) in hash.iter().enumerate() {
            let t = lo + i;
            let mut at = head.get(&h).copied().unwrap_or(NONE);
            while at != NONE {
                if same(at as usize, t) {
                    first = Some((at as usize, t - at as usize));
                    break 'blocks;
                }
                at = next[at as usize];
            }
            next.push(head.insert(h, t as u32).unwrap_or(NONE));
        }
        (lo, block) = (hi, block * 2);
    }
    let conflict = first.and_then(|(mu, lam)| {
        let mut end = n;
        for col in cols {
            if let Some(i) = mismatch(&col[mu + lam..end], &col[mu..end - lam]) {
                end = mu + lam + i;
            }
        }
        (end < n).then(|| end - lam - 1)
    });
    Period { first, conflict }
}

/// The first index where `a` and `b` differ, compared a block at a time.
fn mismatch(a: &[u8], b: &[u8]) -> Option<usize> {
    const BLOCK: usize = 64;
    let blocks = a.chunks(BLOCK).zip(b.chunks(BLOCK));
    for (k, (x, y)) in blocks.enumerate() {
        if x != y {
            return x
                .iter()
                .zip(y)
                .position(|(p, q)| p != q)
                .map(|i| k * BLOCK + i);
        }
    }
    None
}
