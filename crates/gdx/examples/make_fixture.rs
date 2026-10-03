//! Writes a small trnsport-like GDX fixture used by the Python test suite.
use gdx::{GdxWriter, Record, SymbolType};
use std::sync::Arc;

fn rec(keys: &[&str], level: f64) -> Record {
    Record {
        keys: keys.iter().map(|s| Arc::from(*s)).collect(),
        values: [level, 0.0, 0.0, 0.0, 0.0],
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: make_fixture <out.gdx>");
    let mut w = GdxWriter::create(&out, "polars-gdx-fixture").unwrap();
    w.write_symbol(
        "i",
        "canning plants",
        1,
        SymbolType::Set,
        0,
        &[rec(&["seattle"], 0.0), rec(&["san-diego"], 0.0)],
    )
    .unwrap();
    w.write_symbol(
        "j",
        "markets",
        1,
        SymbolType::Set,
        0,
        &[
            rec(&["new-york"], 0.0),
            rec(&["chicago"], 0.0),
            rec(&["topeka"], 0.0),
        ],
    )
    .unwrap();
    let dist = [
        rec(&["seattle", "new-york"], 2.5),
        rec(&["seattle", "chicago"], 1.7),
        rec(&["seattle", "topeka"], 1.8),
        rec(&["san-diego", "new-york"], 2.5),
        rec(&["san-diego", "chicago"], 1.8),
        rec(&["san-diego", "topeka"], 1.4),
    ];
    w.write_symbol(
        "d",
        "distance in thousands of miles",
        2,
        SymbolType::Parameter,
        0,
        &dist,
    )
    .unwrap();
    let mut cost = Vec::new();
    for r in &dist {
        cost.push(Record {
            keys: r.keys.clone(),
            values: [r.values[0] * 90.0 / 1000.0, 0.0, 0.0, 0.0, 0.0],
        });
    }
    w.write_symbol(
        "cost",
        "transport cost per case",
        2,
        SymbolType::Parameter,
        0,
        &cost,
    )
    .unwrap();
    w.write_symbol(
        "a",
        "capacity",
        1,
        SymbolType::Parameter,
        0,
        &[rec(&["seattle"], 350.0), rec(&["san-diego"], 600.0)],
    )
    .unwrap();
    w.write_symbol(
        "b",
        "demand",
        1,
        SymbolType::Parameter,
        0,
        &[
            rec(&["new-york"], 325.0),
            rec(&["chicago"], 300.0),
            rec(&["topeka"], 275.0),
        ],
    )
    .unwrap();
    w.write_symbol(
        "x",
        "shipment quantities",
        2,
        SymbolType::Variable,
        0,
        &[
            rec(&["seattle", "new-york"], 50.0),
            rec(&["seattle", "chicago"], 300.0),
            rec(&["seattle", "topeka"], 0.0),
            rec(&["san-diego", "new-york"], 275.0),
            rec(&["san-diego", "chicago"], 0.0),
            rec(&["san-diego", "topeka"], 275.0),
        ],
    )
    .unwrap();
    w.write_symbol(
        "supply",
        "supply limit",
        1,
        SymbolType::Equation,
        0,
        &[rec(&["seattle"], 0.0), rec(&["san-diego"], 0.0)],
    )
    .unwrap();
    w.write_symbol(
        "demand",
        "demand limit",
        1,
        SymbolType::Equation,
        0,
        &[
            rec(&["new-york"], 0.0),
            rec(&["chicago"], 0.0),
            rec(&["topeka"], 0.0),
        ],
    )
    .unwrap();
    w.finish().unwrap();
    println!("wrote {out}");
}
