// Standalone fixture generator using the gdx crate's writer.
use gdx::{GdxWriter, Record, SymbolType};

fn main() {
    let n0 = 10i64; // dim0
    let n1 = 10i64; // dim1
    let n2 = 1000i64; // dim2 (large)
    let n3 = 10i64;
    let n4 = 6i64;
    let total = (n0 * n1 * n2 * n3 * n4) as usize;
    eprintln!("total records: {total}");
    let compr = std::env::var("COMPRESS").is_ok();
    if compr {
        std::env::set_var("GDXCOMPRESS", "1");
    }
    let out = std::env::var("OUT").unwrap_or_else(|_| "/tmp/big.gdx".into());
    let mut w = GdxWriter::create(&out, "polars-gdx bench").unwrap();
    let mut records = Vec::with_capacity(total);
    for a in 1..=n0 {
        for b in 1..=n1 {
            for c in 1..=n2 {
                for d in 1..=n3 {
                    for e in 1..=n4 {
                        let keys = vec![
                            format!("i{a}").into(),
                            format!("j{b}").into(),
                            format!("k{c}").into(),
                            format!("l{d}").into(),
                            format!("m{e}").into(),
                        ];
                        records.push(Record {
                            keys,
                            values: [(a + b + c + d + e) as f64, 0.0, 0.0, 0.0, 0.0],
                        });
                    }
                }
            }
        }
    }
    w.write_symbol_with_domains(
        "big",
        "large 5-dim parameter",
        5,
        SymbolType::Parameter,
        0,
        &records,
        &["i", "j", "k", "l", "m"],
    )
    .unwrap();
    w.finish().unwrap();
    eprintln!("written {out} (compressed={compr})");
}
