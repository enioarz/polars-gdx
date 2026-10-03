//! Relaxed-domain write support: symbols written with explicit domain names
//! must report them back on read (used by the key/value collision fixture).

use std::sync::Arc;

use gdx::{GdxFile, GdxWriter, Record, SymbolType};

fn rec(keys: &[&str], level: f64) -> Record {
    Record {
        keys: keys.iter().map(|s| Arc::from(*s)).collect(),
        values: [level, 0.0, 0.0, 0.0, 0.0],
    }
}

#[test]
fn write_with_domains_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("domains.gdx");
    let mut w = GdxWriter::create(&path, "gdxcomp-test").unwrap();
    w.write_symbol(
        "value",
        "collision-prone domain set",
        1,
        SymbolType::Set,
        0,
        &[rec(&["alpha"], 0.0), rec(&["beta"], 0.0)],
    )
    .unwrap();
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
    w.finish().unwrap();

    let file = GdxFile::open(&path).unwrap();
    let p = file.symbol("p").expect("symbol p");
    assert_eq!(p.dim, 2);
    assert_eq!(p.domains, vec!["value".to_string(), "value".to_string()]);
    let records = file.read("p").unwrap();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].values[0], 1.0);
}
