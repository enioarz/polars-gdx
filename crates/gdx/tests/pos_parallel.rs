//! Parity checks for the positional parallel read. The large-symbol test
//! uses a generated fixture (see `examples/gen_fixture.rs`); it is skipped
//! when the fixture is absent so CI stays hermetic.

use gdx::{
    read_symbol_raw_parallel_pos, read_symbol_raw_span_parallel, GdxFile, RecordAction, ValueField,
};

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

const COMPRESSED_FIXTURE: &str = "/tmp/big_compressed.gdx";

/// Block-compressed symbols use (block start, offset-in-block) checkpoints:
/// the positional parallel read must deliver exactly the serial data.
#[test]
fn pos_parallel_matches_serial_compressed() {
    if !std::path::Path::new(COMPRESSED_FIXTURE).exists() {
        eprintln!("skipping: {COMPRESSED_FIXTURE} not found (generate with COMPRESS=1 OUT={COMPRESSED_FIXTURE} cargo run --release -p gdx --example gen_fixture)");
        return;
    }
    let file = GdxFile::open(COMPRESSED_FIXTURE).unwrap();
    let info = file.symbol("big").expect("symbol big");
    let vf = ValueField::Level;
    let serial = file.read_symbol_raw(info, vf, None, None).unwrap();
    for threads in [2usize, 3, 4, 8] {
        let builder = || accept_all_builder();
        let par =
            read_symbol_raw_parallel_pos(COMPRESSED_FIXTURE, info, vf, &builder, threads).unwrap();
        assert_eq!(par.values.len(), serial.values.len(), "threads={threads}");
        for d in 0..info.dim {
            assert_eq!(par.keys[d], serial.keys[d], "dim {d} threads={threads}");
        }
    }
}

/// Dim-0 span predicate: accept records whose first-dimension UEL number
/// (1-based, as delivered by the raw read) lies in [lo, hi].
#[allow(clippy::type_complexity)]
fn span_pred_builder(lo: i32, hi: i32) -> Box<dyn Fn(&[i32]) -> RecordAction + Send> {
    Box::new(move |keys: &[i32]| {
        let k = keys[0];
        if (lo..=hi).contains(&k) {
            RecordAction::Accept
        } else {
            RecordAction::Skip
        }
    })
}

/// The span-seek read must deliver exactly the records whose dim-0 key lies
/// in the requested span, for spans anywhere in the file (the prefix before
/// the span must not be decoded) and for any worker count.
#[test]
fn span_parallel_matches_serial_filtered() {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("skipping: {FIXTURE} not found (generate with `cargo run --release -p gdx --example gen_fixture`)");
        return;
    }
    let file = GdxFile::open(FIXTURE).unwrap();
    let info = file.symbol("big").expect("symbol big");
    let vf = ValueField::Level;
    let serial = file.read_symbol_raw(info, vf, None, None).unwrap();
    // Distinct dim-0 keys (0-based UEL positions), sorted.
    let mut dim0: Vec<u32> = serial.keys[0].clone();
    dim0.sort_unstable();
    dim0.dedup();
    // Spans: late (exercises the seek past the prefix), middle, single
    // group, whole file, and an empty span (lo > hi).
    let n = dim0.len() as i32;
    let spans: Vec<(i32, i32)> = vec![
        (
            dim0[n as usize - 3] as i32 + 1,
            dim0[n as usize - 2] as i32 + 1,
        ),
        (
            dim0[n as usize / 2] as i32 + 1,
            dim0[n as usize / 2 + 1] as i32 + 1,
        ),
        (dim0[0] as i32 + 1, dim0[0] as i32 + 1),
        (dim0[0] as i32 + 1, *dim0.last().unwrap() as i32 + 1),
        (dim0[0] as i32 + 1, dim0[0] as i32), // empty: lo > hi
    ];
    for (lo, hi) in spans {
        // Expected: serial read filtered post-hoc to the span.
        let expected: Vec<usize> = serial.keys[0]
            .iter()
            .enumerate()
            .filter(|(_, &k)| (lo..=hi).contains(&(k as i32 + 1)))
            .map(|(i, _)| i)
            .collect();
        for threads in [1usize, 2, 3, 4, 8] {
            let builder = || span_pred_builder(lo, hi);
            let span = read_symbol_raw_span_parallel(FIXTURE, info, vf, &builder, threads, lo, hi)
                .unwrap();
            assert_eq!(
                span.values.len(),
                expected.len(),
                "span [{lo},{hi}] threads={threads}: row count"
            );
            for (d, sk) in span.keys.iter().enumerate() {
                let want: Vec<u32> = expected.iter().map(|&i| serial.keys[d][i]).collect();
                assert_eq!(*sk, want, "span [{lo},{hi}] dim {d} threads={threads}");
            }
            let want: Vec<f64> = expected.iter().map(|&i| serial.values[i]).collect();
            assert_eq!(span.values, want, "span [{lo},{hi}] threads={threads}");
        }
    }
}

/// The span-seek read works on block-compressed symbols (checkpoint pairs
/// resume mid-block exactly) and must deliver exactly the in-span records.
#[test]
fn span_parallel_matches_serial_filtered_compressed() {
    if !std::path::Path::new(COMPRESSED_FIXTURE).exists() {
        eprintln!("skipping: {COMPRESSED_FIXTURE} not found (generate with COMPRESS=1 OUT={COMPRESSED_FIXTURE} cargo run --release -p gdx --example gen_fixture)");
        return;
    }
    let file = GdxFile::open(COMPRESSED_FIXTURE).unwrap();
    let info = file.symbol("big").expect("symbol big");
    let vf = ValueField::Level;
    let serial = file.read_symbol_raw(info, vf, None, None).unwrap();
    let (lo, hi) = (3i32, 3i32);
    let expected: Vec<usize> = serial.keys[0]
        .iter()
        .enumerate()
        .filter(|(_, &k)| (lo..=hi).contains(&(k as i32 + 1)))
        .map(|(i, _)| i)
        .collect();
    for threads in [1usize, 2, 4] {
        let builder = || span_pred_builder(lo, hi);
        let span =
            read_symbol_raw_span_parallel(COMPRESSED_FIXTURE, info, vf, &builder, threads, lo, hi)
                .unwrap();
        assert_eq!(
            span.values.len(),
            expected.len(),
            "threads={threads}: row count"
        );
        let want: Vec<f64> = expected.iter().map(|&i| serial.values[i]).collect();
        assert_eq!(span.values, want, "threads={threads}");
    }
}
