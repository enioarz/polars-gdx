//! Parity checks for the positional parallel read. The large-symbol test
//! uses a generated fixture (see `examples/gen_fixture.rs`); it is skipped
//! when the fixture is absent so CI stays hermetic.

use gdx::{read_symbol_raw_parallel_pos, GdxFile, RecordAction, ValueField};

const FIXTURE: &str = "/tmp/big.gdx";

#[allow(clippy::type_complexity)]
fn accept_all_builder() -> Box<dyn Fn(&[i32]) -> RecordAction + Send> {
    Box::new(|_keys: &[i32]| RecordAction::Accept)
}

#[test]
fn pos_parallel_matches_serial() {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("skipping: {FIXTURE} not found (generate with `cargo run --release -p gdx --example gen_fixture`)");
        return;
    }
    let file = GdxFile::open(FIXTURE).unwrap();
    let info = file.symbol("big").expect("symbol big");
    let vf = ValueField::Level;
    let serial = file.read_symbol_raw(info, vf, None, None).unwrap();
    for threads in [2usize, 3, 4, 6, 8, 16] {
        let builder = || accept_all_builder();
        let par = read_symbol_raw_parallel_pos(FIXTURE, info, vf, &builder, threads).unwrap();
        assert_eq!(par.values.len(), serial.values.len(), "threads={threads}");
        for d in 0..info.dim {
            assert_eq!(par.keys[d], serial.keys[d], "dim {d} threads={threads}");
        }
    }
}

#[test]
fn pos_parallel_repeats_are_deterministic() {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("skipping: {FIXTURE} not found (generate with `cargo run --release -p gdx --example gen_fixture`)");
        return;
    }
    let file = GdxFile::open(FIXTURE).unwrap();
    let info = file.symbol("big").expect("symbol big");
    let vf = ValueField::Level;
    let builder = || accept_all_builder();
    let a = read_symbol_raw_parallel_pos(FIXTURE, info, vf, &builder, 4).unwrap();
    let b = read_symbol_raw_parallel_pos(FIXTURE, info, vf, &builder, 4).unwrap();
    assert_eq!(a.values, b.values);
    for d in 0..info.dim {
        assert_eq!(a.keys[d], b.keys[d]);
    }
}
