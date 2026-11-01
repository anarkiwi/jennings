//! Run a 64 KiB image under `run_until` with a periodic IRQ and report Mcycles/s.
//!
//! `cargo run --release --example bench -- IMAGE [PC] [CYCLES] [IRQ_PERIOD]`
//! (`tools/bench.py --dump IMAGE` writes the playroutine image it benchmarks).

use std::time::Instant;

use jennings_core::{run_until, Cpu, FlatMemory, NoHooks, Periodic, Until};

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i)
        .map(|s| {
            let s = s.trim_start_matches("0x");
            s.parse().ok().unwrap_or_else(|| panic!("bad argument {s}"))
        })
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image = std::fs::read(args.get(1).expect("IMAGE")).expect("read IMAGE");
    assert_eq!(image.len(), 0x10000, "image must be 64 KiB");
    let pc = u16::from_str_radix(
        args.get(2).map_or("1000", |s| s.trim_start_matches("0x")),
        16,
    )
    .expect("PC in hex");
    let cycles: u64 = arg(&args, 3, 1_000_000_000);
    let period: u64 = arg(&args, 4, 20_000);

    let mem = FlatMemory::new();
    for (a, &b) in image.iter().enumerate() {
        mem.set_ram(a as u16, b);
    }
    mem.set_io(true);
    mem.set_raster(Some((63, 312)));
    let mut cpu = Cpu::new();
    cpu.pc = pc;
    let mut hooks = NoHooks;
    let mut bus = mem.bus(&mut hooks);
    let mut until = Until {
        max_cycles: Some(cycles - 1),
        irq: Some(Periodic {
            period,
            due: period,
        }),
        ..Until::default()
    };
    let t0 = Instant::now();
    let stop = run_until(&mut cpu, &mut bus, &mut until, || false);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{stop:?}: {} cycles, {} instructions, {:.2} Mcycles/s, {:.2} ns/instruction, \
         pc={:04X} a={:02X} x={:02X} y={:02X} sp={:02X} p={:02X}",
        cpu.cycles,
        cpu.steps,
        cpu.cycles as f64 / dt / 1e6,
        dt * 1e9 / cpu.steps as f64,
        cpu.pc,
        cpu.a,
        cpu.x,
        cpu.y,
        cpu.sp,
        cpu.p()
    );
}
