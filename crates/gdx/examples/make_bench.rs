//! Generates a large GDX benchmark file.
//!
//! Symbols:
//! - `bigpar`: 2-dim parameter, N_ROWS records (default 2,000,000)
//! - `bigvar`: 2-dim variable over the same domain
use std::sync::Arc;

use gdx::{GdxWriter, Record, SymbolType};

fn rec(keys: &[&str], level: f64) -> Record {
    Record {
        keys: keys.iter().map(|s| Arc::from(*s)).collect(),
        values: [level, 0.0, 0.0, 0.0, 0.0],
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: make_bench <out.gdx> [n_rows]");
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);

    // Domain: n / 1000 first-dim keys x 1000 second-dim keys.
    let d1 = (n / 1000).max(1);
    let d2 = n / d1.max(1);
    println!("writing {d1} x {d2} = {} records", d1 * d2);

    let mut w = GdxWriter::create(&out, "polars-gdx-bench").unwrap();
    let i_keys: Vec<String> = (0..d1).map(|i| format!("i{i:05}")).collect();
    let j_keys: Vec<String> = (0..d2).map(|j| format!("j{j:05}")).collect();
    w.write_symbol(
        "i",
        "first dimension",
        1,
        SymbolType::Set,
        0,
        &i_keys.iter().map(|k| rec(&[k], 0.0)).collect::<Vec<_>>(),
    )
    .unwrap();
    w.write_symbol(
        "j",
        "second dimension",
        1,
        SymbolType::Set,
        0,
        &j_keys.iter().map(|k| rec(&[k], 0.0)).collect::<Vec<_>>(),
    )
    .unwrap();

    let mut par: Vec<Record> = Vec::with_capacity(d1 * d2);
    let mut var: Vec<Record> = Vec::with_capacity(d1 * d2);
    for (a, i) in i_keys.iter().enumerate() {
        for (b, j) in j_keys.iter().enumerate() {
            let v = (a as f64) * 0.5 + (b as f64) * 0.25;
            par.push(rec(&[i, j], v));
            var.push(Record {
                keys: par.last().unwrap().keys.clone(),
                values: [v, v * 0.1, 0.0, 1e5, 1.0],
            });
        }
    }
    w.write_symbol(
        "bigpar",
        "large parameter",
        2,
        SymbolType::Parameter,
        0,
        &par,
    )
    .unwrap();
    w.write_symbol("bigvar", "large variable", 2, SymbolType::Variable, 0, &var)
        .unwrap();
    w.finish().unwrap();
    println!("wrote {out}");
}
