use std::cell::RefCell;

use jennings_core::{
    run_to_rts, run_until, Access, AddrSet, Bus, Cpu, FlatMemory, Hooks, NoHooks, Periodic, Stop,
    Until,
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
    let r = run_to_rts(&mut c, &mut *m, 0x2000, 9, None, 100, || false);
    assert_eq!((r, c.pc, c.a, c.sp), (Stop::Sp, 0x0002, 9, 0xF0));
    let r = run_to_rts(&mut c, &mut *m, 0x2000, 9, Some(&pcs), 100, || false);
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
    assert_eq!(m.last_load(), 0xC000);
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
