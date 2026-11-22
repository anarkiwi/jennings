use std::cell::RefCell;

use jennings_core::{
    access_key, orbit, run_to_rts, run_until, Access, Accesses, AddrSet, Bus, Cpu, End, FlatMemory,
    Hooks, NoHooks, Periodic, Remap, Stop, Until, AS_IS, ZERO,
};

fn ram(prog: &[(u16, &[u8])]) -> Box<[u8; 0x10000]> {
    let mut m = Box::new([0u8; 0x10000]);
    for (org, code) in prog {
        m[*org as usize..*org as usize + code.len()].copy_from_slice(code);
    }
    m
}

#[test]
fn step_charges_page_cross_and_branch_extras() {
    // LDA $10FF,X (X=1, crosses) ; BNE -$80 from $1003 (taken, crosses to $0F85) ; LDA ($10),Y
    let mut m = ram(&[(0x1000, &[0xBD, 0xFF, 0x10, 0xD0, 0x80])]);
    let mut c = Cpu::new();
    c.pc = 0x1000;
    c.x = 1;
    m[0x1100] = 7;
    assert_eq!(c.step(&mut *m), 5);
    assert_eq!((c.a, c.z), (7, 0));
    assert_eq!(c.step(&mut *m), 4);
    assert_eq!(c.pc, 0x0F85);
    m[0x0F85..0x0F87].copy_from_slice(&[0xB1, 0x10]);
    m[0x10..0x12].copy_from_slice(&[0xF0, 0x20]);
    c.y = 0x20;
    assert_eq!(c.step(&mut *m), 6);
    assert_eq!(c.cycles, 15);
}

#[test]
fn interrupts_jam_and_reset() {
    let mut m = ram(&[
        (0xFFFA, &[0x00, 0x50, 0x00, 0x40, 0x00, 0x30]),
        (0x1000, &[0x02]),
    ]);
    let mut c = Cpu::new();
    c.reset(&mut *m, None);
    assert_eq!(c.pc, 0x4000);
    c.pc = 0x1000;
    assert_eq!(c.step(&mut *m), 2);
    assert!(c.jammed && c.jams == 1 && c.pc == 0x1001);
    assert_eq!(c.irq(&mut *m), 7);
    assert_eq!((c.pc, c.sp, m[0x1FD], c.i), (0x3000, 0xFC, 0x20, 1));
    assert_eq!(c.irq(&mut *m), 0);
    assert_eq!(c.nmi(&mut *m), 7);
    assert_eq!(c.pc, 0x5000);
    c.set_p(0xFF);
    assert_eq!(c.p(), 0xFF);
    c.set_p(0x00);
    assert_eq!(c.p(), 0x30);
    assert_eq!(Cpu::default(), Cpu::new());
}

#[test]
fn drivers_stop_where_asked() {
    // JSR $2000 ; JAM ; ... $2000: INX ; RTS
    let mut m = ram(&[(0x1000, &[0x20, 0x00, 0x20, 0x02]), (0x2000, &[0xE8, 0x60])]);
    let mut c = Cpu::new();
    c.pc = 0x1000;
    let pcs: AddrSet = [0x2001].into_iter().collect();
    let mut u = Until {
        stop_pcs: Some(&pcs),
        ..Until::default()
    };
    assert_eq!(run_until(&mut c, &mut *m, &mut u, || false), Stop::Pc);
    let mut u = Until {
        stop_on_jam: true,
        ..Until::default()
    };
    assert_eq!(run_until(&mut c, &mut *m, &mut u, || false), Stop::Jam);
    let mut u = Until {
        max_cycles: Some(1),
        ..Until::default()
    };
    assert_eq!(run_until(&mut c, &mut *m, &mut u, || false), Stop::Cycles);
    assert_eq!(
        run_until(&mut c, &mut *m, &mut Until::default(), || true),
        Stop::Stopped
    );
    c.sp = 0xF0;
    let r = run_to_rts(&mut c, &mut *m, 0x2000, 9, None, 100, None, || false);
    assert_eq!((r, c.pc, c.a, c.sp), (Stop::Sp, 0x0002, 9, 0xF0));
    let r = run_to_rts(&mut c, &mut *m, 0x2000, 9, Some(&pcs), 100, None, || false);
    assert_eq!((r, c.pc), (Stop::Pc, 0x2001));
}

#[derive(Default)]
struct Log(RefCell<Vec<(char, u16, u8, Access)>>);

impl Hooks for &Log {
    fn read_hit(&mut self, _m: &FlatMemory, a: u16, k: Access) -> Option<u8> {
        self.0.borrow_mut().push(('r', a, 0, k));
        (a == 0xC000).then_some(0x42)
    }
    fn write_hit(&mut self, m: &FlatMemory, a: u16, v: u8, k: Access) -> u8 {
        self.0.borrow_mut().push(('w', a, v, k));
        m.set_page(0xE, None);
        v ^ 0xFF
    }
    fn observe_load(&mut self, _m: &FlatMemory, a: u16) {
        self.0.borrow_mut().push(('l', a, 0, Access::Data));
    }
    fn observe_store(&mut self, _m: &FlatMemory, a: u16, v: u8) {
        self.0.borrow_mut().push(('s', a, v, Access::Data));
    }
}

#[test]
fn flat_memory_pages_watches_observers() {
    let m = FlatMemory::default();
    m.set_page(0xE, Some(&[0xEA; 0x1000]));
    assert!(m.page_mapped(0xE));
    // LDA $C000 ; STA $E000 ; PHA ; JMP ($C0FF)
    for (i, b) in [0xAD, 0x00, 0xC0, 0x8D, 0x00, 0xE0, 0x48, 0x6C, 0xFF, 0xC0]
        .iter()
        .enumerate()
    {
        m.set_ram(0x1000 + i as u16, *b);
    }
    m.watch_read(0xC000, true);
    m.watch_read(0xC0FF, true);
    m.watch_write(0xE000, true);
    m.observe_load(0xC000, true);
    m.observe_store(0xE000, true);
    m.watch_write(0x01FF, true);
    let log = Log::default();
    let mut h = &log;
    let mut c = Cpu::new();
    c.pc = 0x1000;
    {
        let mut bus = m.bus(&mut h);
        for _ in 0..4 {
            c.step(&mut bus);
        }
    }
    assert_eq!(m.last_load(), Some(0xC000));
    assert_eq!(m.ram(0xE000), 0x42 ^ 0xFF);
    assert!(!m.page_mapped(0xE));
    assert_eq!(
        *log.0.borrow(),
        vec![
            ('r', 0xC000, 0, Access::Data),
            ('l', 0xC000, 0, Access::Data),
            ('w', 0xE000, 0x42, Access::Data),
            ('s', 0xE000, 0x42, Access::Data),
            ('w', 0x01FF, 0x42, Access::Stack),
            ('r', 0xC0FF, 0, Access::Other),
            ('r', 0xC000, 0, Access::Other),
        ]
    );
    assert_eq!(m.peek(&mut NoHooks, 0xC000), 0);
    m.poke(&mut NoHooks, 0xE000, 5);
    assert_eq!(m.ram(0xE000), 5);
}

#[test]
fn io_page_and_raster() {
    let m = FlatMemory::new();
    m.set_ram(0xD011, 0x1B);
    m.set_raster(Some((63, 312)));
    assert_eq!(m.raster(), Some((63, 312)));
    m.set_now(63 * 300);
    assert_eq!(m.peek(&mut NoHooks, 0xD012), 0);
    m.set_io(true);
    assert!(m.io());
    assert_eq!(m.now(), 63 * 300);
    assert_eq!(m.peek(&mut NoHooks, 0xD012), (300 & 0xFF) as u8);
    assert_eq!(m.peek(&mut NoHooks, 0xD011), 0x9B);
    assert_eq!(m.peek(&mut NoHooks, 0xD020), 0);
    m.poke(&mut NoHooks, 0xD400, 9);
    assert_eq!(m.ram(0xD400), 9);
    m.set_raster(None);
    assert_eq!(m.peek(&mut NoHooks, 0xD012), 0);
}

#[test]
fn bus_defaults_route_to_read_write() {
    struct Plain([u8; 0x10000]);
    impl Bus for Plain {
        fn read(&mut self, a: u16) -> u8 {
            self.0[a as usize]
        }
        fn write(&mut self, a: u16, v: u8) {
            self.0[a as usize] = v
        }
    }
    let mut b = Plain([0; 0x10000]);
    b.store(1, 2);
    b.stack_write(0x1FF, 3);
    assert_eq!((b.load(1), b.stack_read(0x1FF), b.fetch(1)), (2, 3, 2));
    b.sync(&Cpu::new());
}

#[test]
fn periodic_irq_fires_on_schedule_and_drops_when_masked() {
    // $1000: JMP $1000 ; IRQ handler $3000: RTI
    let mut m = ram(&[
        (0x1000, &[0x4C, 0x00, 0x10]),
        (0xFFFE, &[0x00, 0x30]),
        (0x3000, &[0x40]),
    ]);
    let mut c = Cpu::new();
    c.pc = 0x1000;
    let p = Periodic {
        period: 30,
        due: 30,
    };
    let mut u = Until {
        max_cycles: Some(95),
        irq: Some(p),
        ..Until::default()
    };
    run_until(&mut c, &mut *m, &mut u, || false);
    assert_eq!(
        u.irq,
        Some(Periodic {
            period: 30,
            due: 120
        })
    );
    c.pc = 0x1000;
    c.sp = 0xFF;
    c.i = 1;
    let sp = c.sp;
    let mut u = Until {
        max_cycles: Some(30),
        irq: Some(p),
        ..Until::default()
    };
    run_until(&mut c, &mut *m, &mut u, || false);
    assert_eq!((c.sp, c.pc & 0xF000), (sp, 0x1000));
}

#[test]
fn idle_and_irq_stops() {
    // NOP ; JMP $1001 (idles) ; CLI at $2000
    let mut m = ram(&[(0x1000, &[0xEA, 0x4C, 0x01, 0x10]), (0x2000, &[0x58, 0xEA])]);
    let mut c = Cpu::new();
    c.pc = 0x1000;
    let mut u = Until {
        stop_on_idle: true,
        ..Until::default()
    };
    assert_eq!(run_until(&mut c, &mut *m, &mut u, || false), Stop::Idle);
    assert_eq!(c.pc, 0x1001);
    c.pc = 0x2000;
    c.i = 1;
    let mut u = Until {
        stop_on_irq: true,
        ..Until::default()
    };
    assert_eq!(run_until(&mut c, &mut *m, &mut u, || false), Stop::Irq);
    assert_eq!(c.pc, 0x2001);
}

fn observed_by_bus(m: &FlatMemory) -> (Vec<u16>, Vec<u16>) {
    let log = Log::default();
    let mut h = &log;
    let mut bus = m.bus(&mut h);
    for a in 0..=0xFFFFu16 {
        bus.load(a);
        bus.store(a, 0);
    }
    let (mut l, mut s) = (Vec::new(), Vec::new());
    for &(k, a, _, _) in log.0.borrow().iter() {
        match k {
            'l' => l.push(a),
            's' => s.push(a),
            _ => {}
        }
    }
    (l, s)
}

#[test]
fn observe_ranges_replace_and_match_per_address() {
    let cases: &[&[(u32, u32)]] = &[
        &[(0xE000, 0x10000)],
        &[(0x0000, 0x10000)],
        &[(0x003F, 0x0041), (0x0100, 0x0101), (0xD400, 0xD419)],
        &[(0x0005, 0x0005), (0x0080, 0x0040), (0xFFFF, 0x20000)],
        &[(0x0010, 0x0050), (0x0030, 0x0070)],
        &[],
    ];
    let m = FlatMemory::default();
    for (i, ranges) in cases.iter().enumerate() {
        let stores = cases[(i + 1) % cases.len()];
        m.observe_loads(ranges.iter().copied());
        m.observe_stores(stores.iter().copied());
        let want = |rs: &[(u32, u32)]| -> Vec<u16> {
            let reference = FlatMemory::default();
            for &(lo, hi) in rs {
                (lo..hi.min(0x10000)).for_each(|a| reference.observe_load(a as u16, true));
            }
            observed_by_bus(&reference).0
        };
        assert_eq!(observed_by_bus(&m), (want(ranges), want(stores)), "{i}");
    }
}

#[test]
fn accesses_log_data_and_redirect_sites() {
    // $1000 LDA $2005,X ; $1003 STA $2105,X ; $1006 LDA ($10),Y ; $1008 INC $2205 ; $100B PHA ; $100C PLA ; $100D RTS
    let mut m = ram(&[(
        0x1000,
        &[
            0xBD, 0x05, 0x20, 0x9D, 0x05, 0x21, 0xB1, 0x10, 0xEE, 0x05, 0x22, 0x48, 0x68, 0x60,
        ],
    )]);
    m[0x10..0x12].copy_from_slice(&[0x00, 0x23]);
    m[0x2005] = 9;
    m[0x2003] = 4;
    m[0x2205] = 7;
    m[0x2300] = 6;
    let table = |f: &dyn Fn(usize) -> i32| (0..0x10000).map(f).collect::<Vec<i32>>();
    let mut r = Remap::default();
    r.insert(
        0x1000,
        &table(&|a| if a == 0x2005 { 0x2003 } else { AS_IS }),
    );
    r.insert(0x1003, &table(&|a| if a == 0x2105 { ZERO } else { AS_IS }));
    r.insert(
        0x1006,
        &table(&|a| {
            if a < 0x100 || a == 0x2300 {
                ZERO
            } else {
                AS_IS
            }
        }),
    );
    r.insert(
        0x1008,
        &table(&|a| if a == 0x2205 { 0x2003 } else { AS_IS }),
    );
    r.insert(
        0x100B,
        &table(&|a| {
            if (0x100..0x200).contains(&a) {
                ZERO
            } else {
                AS_IS
            }
        }),
    );
    r.set_pointer_opcodes([0xB1]);
    let mut c = Cpu::new();
    let mut bus = Accesses::new(&mut *m, Some(&r));
    run_to_rts(&mut c, &mut bus, 0x1000, 0, None, 1000, None, || false);
    let log = bus.log;
    assert_eq!(
        log,
        [
            access_key(0x1000, false, 0x2005),
            access_key(0x1003, true, 0x2105),
            access_key(0x1006, false, 0x10),
            access_key(0x1006, false, 0x11),
            access_key(0x1006, false, 0x2300),
            access_key(0x1008, false, 0x2205),
            access_key(0x1008, true, 0x2205),
        ]
    );
    // The load read $2003, the store, the INC and PHA's push wrote nothing; PLA pulled 0.
    assert_eq!((m[0x2105], m[0x2205], m[0x1FD], c.a), (0, 7, 0, 0));
    let mut c = Cpu::new();
    let mut bus = Accesses::new(&mut *m, None);
    run_to_rts(&mut c, &mut bus, 0x1000, 0, None, 1000, None, || false);
    assert_eq!(bus.log, log);
    assert_eq!((m[0x2105], m[0x2205], m[0x1FD], c.a), (9, 8, 6, 6));
}

#[test]
fn remap_replaces_a_table() {
    let mut m = ram(&[(0x1000, &[0xAD, 0x05, 0x20, 0x60])]);
    m[0x2005] = 1;
    m[0x2006] = 2;
    let mut r = Remap::default();
    let to = |s: i32| {
        let mut t = vec![AS_IS; 0x10000];
        t[0x2005] = s;
        t
    };
    r.insert(0x1000, &to(ZERO));
    r.insert(0x1000, &to(0x2006));
    let mut c = Cpu::new();
    run_to_rts(
        &mut c,
        &mut Accesses::new(&mut *m, Some(&r)),
        0x1000,
        0,
        None,
        100,
        None,
        || false,
    );
    assert_eq!(c.a, 2);
}

#[test]
fn remap_keeps_zpx_pointer_bytes_indexed_by_x() {
    // $1000 LDX #$04 ; $1002 LDA ($10,X) ; $1004 RTS: the pointer is $14/$15, not $10/$11.
    let mut m = ram(&[(0x1000, &[0xA2, 0x04, 0xA1, 0x10, 0x60])]);
    m[0x14..0x16].copy_from_slice(&[0x10, 0x00]);
    m[0x2014..0x2016].copy_from_slice(&[0x10, 0x00]);
    m[0x10] = 9;
    m[0x2010] = 7;
    let t: Vec<i32> = (0..0x10000)
        .map(|a| if a < 0x100 { a + 0x2000 } else { AS_IS })
        .collect();
    let mut r = Remap::default();
    r.insert(0x1002, &t);
    r.set_pointer_opcodes([0xA1]);
    let mut c = Cpu::new();
    run_to_rts(
        &mut c,
        &mut Accesses::remapped(&mut *m, &r),
        0x1000,
        0,
        None,
        100,
        None,
        || false,
    );
    assert_eq!(c.a, 7);
}

#[test]
fn remapped_redirects_without_logging() {
    let prog: &[u8] = &[0xAD, 0x05, 0x20, 0x8D, 0x07, 0x20, 0x60];
    let mut t = vec![AS_IS; 0x10000];
    t[0x2005] = 0x2006;
    t[0x2007] = ZERO;
    let mut r = Remap::default();
    r.insert(0x1000, &t);
    r.insert(0x1003, &t);
    let run = |remapped: bool| {
        let mut m = ram(&[(0x1000, prog)]);
        m[0x2005] = 1;
        m[0x2006] = 2;
        let mut c = Cpu::new();
        let mut bus = if remapped {
            Accesses::remapped(&mut *m, &r)
        } else {
            Accesses::new(&mut *m, Some(&r))
        };
        run_to_rts(&mut c, &mut bus, 0x1000, 0, None, 100, None, || false);
        let n = bus.log.len();
        (c.a, m[0x2007], n)
    };
    assert_eq!(run(false), (2, 0, 2));
    assert_eq!(run(true), (2, 0, 0));
}

#[test]
fn digest_covers_the_spans_only() {
    let m = FlatMemory::new();
    let all = [(0u32, 0x10000u32)];
    let parts = [(0u32, 0x100u32), (0x200, 0x10000)];
    let (d0, p0) = (m.digest(all), m.digest(parts));
    m.set_ram(0x150, 1);
    assert_eq!(m.digest(parts), p0);
    assert_ne!(m.digest(all), d0);
    m.set_ram(0x150, 0);
    assert_eq!(m.digest(all), d0);
    for a in [0u16, 0x7, 0x1FF0, 0xFFFF] {
        m.set_ram(a, 0x80);
        assert_ne!(m.digest(all), d0, "{a:#x}");
        m.set_ram(a, 0);
    }
    assert_ne!(m.digest([(0, 0x8000), (0x8000, 0x10000)]), d0);
    assert_eq!(m.digest([(0x20000, 0x30000), (5, 5)]), m.digest([]));
}

/// $1000 INC $10 ; LDA $10 ; CMP #3 ; BNE $100C ; LDA #0 ; STA $10 ; $100C RTS: $10 counts 0, 1, 2.
fn counter() -> Box<[u8; 0x10000]> {
    ram(&[(
        0x1000,
        &[
            0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00, 0x85, 0x10, 0x60,
        ],
    )])
}

fn zero_page(b: &mut [u8; 0x10000]) -> u128 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut h = DefaultHasher::new();
    b[..0x100].hash(&mut h);
    h.finish() as u128
}

#[test]
fn orbit_closes_at_the_first_repeated_state() {
    let (mut m, mut c) = (counter(), Cpu::new());
    let none = AddrSet::default();
    let o = orbit(
        &mut c,
        &mut *m,
        0x1000,
        0,
        1000,
        100,
        None,
        &none,
        zero_page,
        || false,
    );
    assert_eq!((o.end, o.states, m[0x10]), (End::Closed, 3, 0));
    assert_eq!(
        o.accesses,
        [
            access_key(0x1000, false, 0x10),
            access_key(0x1000, true, 0x10),
            access_key(0x1002, false, 0x10),
            access_key(0x100A, true, 0x10),
        ]
    );
    let edge = |a: u64, b: u64| a << 16 | b;
    assert_eq!(
        o.edges,
        [
            edge(0x1000, 0x1002),
            edge(0x1002, 0x1004),
            edge(0x1004, 0x1006),
            edge(0x1006, 0x1008),
            edge(0x1006, 0x100C),
            edge(0x1008, 0x100A),
            edge(0x100A, 0x100C),
        ]
    );
}

#[test]
fn orbit_ends_at_the_limit_a_volatile_load_or_a_jam() {
    let none = AddrSet::default();
    let run = |m: &mut [u8; 0x10000], limit, vol: &AddrSet| {
        orbit(
            &mut Cpu::new(),
            m,
            0x1000,
            0,
            1000,
            limit,
            None,
            vol,
            zero_page,
            || false,
        )
    };
    let mut m = counter();
    let o = run(&mut m, 2, &none);
    assert_eq!((o.end, o.states, m[0x10]), (End::Open, 3, 2));
    assert!(!o.accesses.contains(&access_key(0x100A, true, 0x10)));
    let mut m = counter();
    assert_eq!(run(&mut m, 3, &none).end, End::Closed);
    let mut vol = AddrSet::default();
    vol.insert(0x10);
    let mut m = counter();
    let o = run(&mut m, 100, &vol);
    assert_eq!((o.end, o.states, m[0x10]), (End::Volatile, 1, 1));
    let mut m = ram(&[(0x1000, &[0x02])]);
    assert_eq!(run(&mut m, 100, &none).end, End::Jam);
    let mut m = counter();
    let o = orbit(
        &mut Cpu::new(),
        &mut *m,
        0x1000,
        0,
        1000,
        100,
        None,
        &none,
        zero_page,
        || true,
    );
    assert_eq!((o.end, o.states), (End::Stopped, 1));
}

#[test]
fn orbit_runs_under_the_remap() {
    let mut r = Remap::default();
    let mut t = vec![AS_IS; 0x10000];
    t[0x10] = ZERO;
    r.insert(0x1000, &t);
    let mut m = counter();
    let o = orbit(
        &mut Cpu::new(),
        &mut *m,
        0x1000,
        0,
        1000,
        100,
        Some(&r),
        &AddrSet::default(),
        zero_page,
        || false,
    );
    // INC reads 0 and stores nothing: every call starts from the same state.
    assert_eq!((o.end, o.states, m[0x10]), (End::Closed, 1, 0));
    assert!(o.accesses.contains(&access_key(0x1000, true, 0x10)));
}
