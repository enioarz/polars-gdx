//! Writes fixtures for edge-case tests: a symbol whose domain set is
//! literally named `value` (key/value column-name collision) and mixed-case
//! UELs (case-insensitive label matching).

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
        .expect("usage: make_edgecase_fixture <out.gdx>");
    let mut w = GdxWriter::create(&out, "polars-gdx-fixture").unwrap();

    // Domain sets. The first dimension's domain is literally named "value",
    // so a naive key-naming scheme would collide with the value column.
    w.write_symbol(
        "value",
        "collision-prone domain set",
        1,
        SymbolType::Set,
        0,
        &[rec(&["alpha"], 0.0), rec(&["beta"], 0.0)],
    )
    .unwrap();

    // Parameter over that domain: column names would be ["value", "value"].
    w.write_symbol_with_domains(
        "p",
        "parameter over the value domain",
        2,
        SymbolType::Parameter,
        0,
        &[
            rec(&["alpha", "alpha"], 1.0),
            rec(&["alpha", "beta"], 2.0),
            rec(&["beta", "alpha"], 3.0),
            rec(&["beta", "beta"], 4.0),
        ],
        &["value", "value"],
    )
    .unwrap();

    // Mixed-case UELs for case-insensitive matching tests.
    w.write_symbol(
        "mixed",
        "mixed-case labels",
        1,
        SymbolType::Parameter,
        0,
        &[
            rec(&["Seattle"], 10.0),
            rec(&["SAN-DIEGO"], 20.0),
            rec(&["topeka"], 30.0),
        ],
    )
    .unwrap();

    w.finish().unwrap();
    println!("wrote {out}");
}
