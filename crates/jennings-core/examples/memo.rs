//! Time [`memo_orbit`] of `PC` on 64 KiB images, `$D000-$DFFF` volatile and the rest the
//! state, at each of `GRAINS` (`call`, `sub`, `loop`, `split`), `FIBRE` (hex addresses joined
//! by commas, `-` for none) set apart; with `brute` check each against [`orbit`].
//!
//! `cargo run --release --example memo -- PC LIMIT GRAINS FIBRE [brute] IMAGE...`

use std::time::Instant;

use std::hash::{DefaultHasher, Hash, Hasher};

use jennings_core::{memo_orbit, memo_orbit_fibre, orbit, AddrSet, Cpu, Grain};

fn image(data: &[u8]) -> Box<[u8; 0x10000]> {
    let mut ram = Box::new([0u8; 0x10000]);
    ram.copy_from_slice(data);
    ram
}

fn digest(_: &Cpu, ram: &mut [u8; 0x10000]) -> u128 {
    let (mut lo, mut hi) = (DefaultHasher::new(), DefaultHasher::new());
    (&ram[..0xD000], &ram[0xE000..]).hash(&mut lo);
    (&ram[0xE000..], 1u8, &ram[..0xD000]).hash(&mut hi);
    (hi.finish() as u128) << 64 | lo.finish() as u128
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pc = u16::from_str_radix(args[1].trim_start_matches("0x"), 16).expect("PC in hex");
    let limit: usize = args[2].parse().expect("LIMIT");
    let grains: Vec<Grain> = args[3]
        .split(',')
        .map(|g| match g {
            "call" => Grain::Call,
            "sub" => Grain::Sub,
            "loop" => Grain::Loop,
            "split" => Grain::Split,
            _ => panic!("GRAINS are call, sub, loop, split"),
        })
        .collect();
    let cells = args[4].split(',').filter(|c| *c != "-");
    let fibre: Vec<u16> = cells
        .map(|c| u16::from_str_radix(c, 16).expect("FIBRE in hex"))
        .collect();
    let brute = args.get(5).is_some_and(|a| a == "brute");
    let (mut volatile, mut keep) = (AddrSet::default(), AddrSet::default());
    volatile.insert_range(0xD000, 0xE000);
    keep.insert_range(0, 0xD000);
    keep.insert_range(0xE000, 0x10000);
    for path in &args[5 + brute as usize..] {
        let data = std::fs::read(path).expect("read IMAGE");
        let name = path.rsplit('/').next().unwrap_or(path);
        let want = brute.then(|| {
            let mut ram = image(&data);
            let (mut cpu, t0) = (Cpu::new(), Instant::now());
            let o = orbit(
                &mut cpu,
                &mut *ram,
                pc,
                0,
                10_000_000,
                limit,
                None,
                &volatile,
                digest,
                || false,
            );
            println!(
                "{name} brute {:?} states {} accesses {} edges {} seconds {:.2}",
                o.end,
                o.states,
                o.accesses.len(),
                o.edges.len(),
                t0.elapsed().as_secs_f64()
            );
            o
        });
        for &grain in &grains {
            let mut ram = image(&data);
            let (mut cpu, t0) = (Cpu::new(), Instant::now());
            let (pc, max, none) = (pc, 10_000_000, None);
            let (ram, vol) = (&mut *ram, &volatile);
            let (o, s) = match fibre.is_empty() {
                true => memo_orbit(
                    &mut cpu,
                    ram,
                    pc,
                    0,
                    max,
                    limit,
                    none,
                    vol,
                    &keep,
                    &[],
                    grain,
                    || false,
                ),
                false => memo_orbit_fibre(
                    &mut cpu,
                    ram,
                    pc,
                    0,
                    max,
                    limit,
                    none,
                    vol,
                    &keep,
                    &[],
                    grain,
                    &fibre,
                    || false,
                ),
            };
            let same = want.as_ref().map(|w| *w == o);
            println!(
                "{name} {grain:?} {:?} states {} accesses {} edges {} seconds {:.2} same {same:?} {s:?}",
                o.end,
                o.states,
                o.accesses.len(),
                o.edges.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
}
