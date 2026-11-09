//! Generate the M7 task-047 acceptance fixtures (docs/m7-adapter-contract
//! §6 acceptance recipes). Usage:
//!
//! ```sh
//! cargo run -p polars-gdx --release --no-default-features --features gdxrs \
//!     --example gen_m7_acceptance -- <kind> <output.gdx> [records]
//! ```
//!
//! Kinds:
//! - `record-raw`: 400,000,000 eighteen-dim records (2^10 x 5^8 unique
//!   ascending key tuples; the GDX v7 encoding is so compact that the
//!   contract's 80M-record recipe stores to 0.88 GiB and FAILS the
//!   physical > 4 GiB gate, so the record fixtures scale to 400M records
//!   to satisfy it; logical size far exceeds 4x the 1 GiB cap).
//! - `record-zlib`: the same records, zlib-compressed (logical size
//!   exceeds 4x the cap; compressed physical bytes may fit RAM).
//! - `metadata`: 72,000,000 UELs with 63-byte labels (> 4 GiB of label
//!   data) and a 10,000,000-record sparse symbol over that universe.
//!
//! Records are deterministic: value = (ordinal % 1000) as f64, keys in
//! registration order, so expected outputs are computable independently.
//! Source recipe and hashes are recorded by the acceptance runner.

use gdxrs::{
    MetadataOptions, SymbolType, Value, WriteCompression, WriteOptions, WriteSymbol, Writer,
};

fn main() {
    let mut args = std::env::args_os();
    let _ = args.next();
    let kind = args
        .next()
        .expect("kind: record-raw | record-zlib | metadata")
        .to_string_lossy()
        .to_string();
    let output = std::path::PathBuf::from(
        args.next()
            .expect("output path")
            .to_string_lossy()
            .to_string(),
    );
    let records: u64 = args
        .next()
        .and_then(|s| s.to_string_lossy().parse().ok())
        .unwrap_or(400_000_000);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let scratch =
        std::env::temp_dir().join(format!("gen-m7-acceptance-{}-{}", kind, std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let mut meta_options = MetadataOptions::new(&scratch);
    meta_options.maximum_temporary_bytes = 64 * 1024 * 1024 * 1024;
    let start = std::time::Instant::now();
    let write_options = WriteOptions {
        compression: if kind == "record-zlib" {
            WriteCompression::Zlib
        } else {
            WriteCompression::None
        },
        ..WriteOptions::default()
    };
    let mut writer = Writer::create(&output, write_options).expect("create writer");
    writer.begin_metadata(&meta_options).unwrap();
    // Record fixtures use seventeen key dimensions with mixed radices
    // 2^10 x 5^7 = 80,000,000 unique ascending tuples (~120 logical
    // bytes per row); metadata uses the contract's 63-byte labels plus
    // terminator (64 logical bytes) over a 72M-UEL universe.
    let radices: Vec<u64> = if kind == "metadata" {
        vec![72_000_000]
    } else {
        // Factor the record count into radices of 2 and 5 so the mixed-
        // radix enumeration produces exactly `records` unique ascending
        // tuples (400M = 2^10 x 5^8; smoke sizes factor the same way).
        let mut rest = records;
        let mut radix_list = Vec::new();
        while rest.is_multiple_of(2) {
            radix_list.push(2);
            rest /= 2;
        }
        while rest.is_multiple_of(5) {
            radix_list.push(5);
            rest /= 5;
        }
        assert_eq!(rest, 1, "record count must factor into 2s and 5s");
        radix_list
    };
    let label_count: u64 = radices.iter().sum();
    let label_width = if kind == "metadata" { 63 } else { 20 };
    // Registration offset of each dimension's first label.
    let mut offsets = Vec::with_capacity(radices.len());
    let mut running = 0u64;
    for radix in &radices {
        offsets.push(running);
        running += radix;
    }
    let labels = (0..label_count).map(|i| {
        let mut label = vec![b'u'; label_width];
        let text = format!("{i:020}");
        label[label_width - 20..].copy_from_slice(text.as_bytes());
        Ok::<Vec<u8>, gdxrs::Error>(label)
    });
    let mut ids = Vec::with_capacity(label_count as usize);
    writer
        .register_uels_fresh(labels, |_ordinal, id| {
            ids.push(id);
            Ok(())
        })
        .unwrap();
    let meta_elapsed = start.elapsed();
    let symbol = writer
        .register_symbol(&WriteSymbol::new(
            b"BIG",
            SymbolType::Parameter,
            radices.len() as u8,
        ))
        .unwrap();
    let mut rows = writer.begin_symbol(symbol, None).unwrap();
    match kind.as_str() {
        "record-raw" | "record-zlib" => {
            // Strictly ascending unique key tuples: record i is the
            // mixed-radix (2^10 x 5^8) representation of i, most
            // significant dimension first.
            let dims = radices.len();
            let mut keys = vec![ids[0]; dims];
            let mut digits = vec![0u64; dims];
            for i in 0..records {
                let mut rest = i;
                for d in (0..dims).rev() {
                    digits[d] = rest % radices[d];
                    rest /= radices[d];
                }
                for d in 0..dims {
                    keys[d] = ids[(offsets[d] + digits[d]) as usize];
                }
                rows.write(&keys, &[Value::NumberBits(((i % 1000) as f64).to_bits())])
                    .unwrap();
            }
        }
        "metadata" => {
            // Sparse ascending use of the huge universe: every 7th UEL
            // (10M * 7 = 70M < 72M, so the sequence never wraps).
            let used = 10_000_000u64;
            for i in 0..used {
                let id = ids[(i * 7) as usize];
                rows.write(
                    std::slice::from_ref(&id),
                    &[Value::NumberBits(((i % 1000) as f64).to_bits())],
                )
                .unwrap();
            }
        }
        _ => unreachable!(),
    }
    let stats = rows.finish().unwrap();
    writer.finish().unwrap();
    let total = start.elapsed();
    let size = std::fs::metadata(&output).unwrap().len();
    println!(
        "kind={kind} labels={label_count} records={} meta={meta_elapsed:?} total={total:?} \\
         file={size} bytes ({:.3} GiB)",
        stats,
        size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
