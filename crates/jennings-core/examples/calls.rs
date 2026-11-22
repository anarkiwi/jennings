//! Time `CALLS` calls of `PC` (each to its RTS) on a 64 KiB image: plain, through a
//! [`Census`], and a census-driven [`Learn`].
//!
//! `cargo run --release --example calls -- IMAGE PC [CALLS]`

use std::time::Instant;

use jennings_core::{run_to_rts, AddrSet, Census, Cpu, Learn};

fn image(path: &str) -> Box<[u8; 0x10000]> {
    let data = std::fs::read(path).expect("read IMAGE");
    let mut ram = Box::new([0u8; 0x10000]);
    ram.copy_from_slice(&data);
    ram
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pc = u16::from_str_radix(args[2].trim_start_matches("0x"), 16).expect("PC in hex");
    let calls: usize = args.get(3).map_or(100_000, |s| s.parse().expect("CALLS"));
    let rate = |t0: Instant| calls as f64 / t0.elapsed().as_secs_f64();
    let (mut ram, mut cpu) = (image(&args[1]), Cpu::new());
    let t0 = Instant::now();
    for _ in 0..calls {
        run_to_rts(&mut cpu, &mut *ram, pc, 0, None, 1_000_000, None, || false);
    }
    println!("plain:  {:.0} calls/s", rate(t0));
    let (mut ram, mut cpu) = (image(&args[1]), Cpu::new());
    let t0 = Instant::now();
    for _ in 0..calls {
        let start = cpu.sp;
        ram[0x100 | cpu.sp as usize] = 0;
        ram[0x100 | cpu.sp.wrapping_sub(1) as usize] = 1;
        cpu.sp = cpu.sp.wrapping_sub(2);
        cpu.a = 0;
        cpu.pc = pc;
        while cpu.sp < start {
            cpu.step(&mut *ram);
        }
    }
    println!("steps:  {:.0} calls/s", rate(t0));
    let (mut ram, mut cpu) = (image(&args[1]), Cpu::new());
    let t0 = Instant::now();
    let stopped = || false;
    for _ in 0..calls {
        let start = cpu.sp;
        ram[0x100 | cpu.sp as usize] = 0;
        ram[0x100 | cpu.sp.wrapping_sub(1) as usize] = 1;
        cpu.sp = cpu.sp.wrapping_sub(2);
        cpu.a = 0;
        cpu.pc = pc;
        let end = cpu.cycles + 1_000_001;
        loop {
            if cpu.sp >= start {
                break;
            }
            if cpu.cycles >= end {
                break;
            }
            let jams = cpu.jams;
            cpu.step(&mut *ram);
            if stopped() || cpu.jams != jams {
                break;
            }
        }
    }
    println!("checks: {:.0} calls/s", rate(t0));
    let (mut ram, mut cpu) = (image(&args[1]), Cpu::new());
    let mut census = Census::new(vec![]);
    let t0 = Instant::now();
    let peek = |b: &mut [u8; 0x10000], a: u16| b[a as usize];
    census.run(
        &mut cpu,
        &mut *ram,
        pc,
        0,
        1_000_000,
        calls,
        None,
        peek,
        || false,
    );
    println!(
        "census: {:.0} calls/s, {} classes",
        rate(t0),
        census.classes()
    );
    let (mut ram, mut cpu) = (image(&args[1]), Cpu::new());
    let mut learn = Learn::new(&[], AddrSet::default(), false);
    let t0 = Instant::now();
    let poke = |b: &mut [u8; 0x10000], a: u16, v: u8| b[a as usize] = v;
    let stop = learn.run_census(
        &mut cpu,
        &mut *ram,
        &census,
        pc,
        0,
        1_000_000,
        calls,
        None,
        poke,
        || false,
    );
    println!("learn:  {:.0} calls/s ({stop:?})", rate(t0));
}
