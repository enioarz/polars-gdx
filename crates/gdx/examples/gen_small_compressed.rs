//! Generates a small block-compressed GDX fixture for the Python test suite.
use gdx::{GdxWriter, Record, SymbolType};
use std::sync::Arc;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: gen_small_compressed <out.gdx>");
    std::env::set_var("GDXCOMPRESS", "1");
    let mut w = GdxWriter::create(&out, "polars-gdx fixture").unwrap();
    let mut records = Vec::new();
    for i in 1..=40i64 {
        for j in 1..=500i64 {
            records.push(Record {
                keys: vec![format!("i{i}").into(), format!("j{j}").into()],
                values: [(i * 1000 + j) as f64, 0.0, 0.0, 0.0, 0.0],
            });
        }
    }
    let _: Arc<str> = Arc::from("x");
    w.write_symbol_with_domains(
        "big",
        "2-dim compressed parameter",
        2,
        SymbolType::Parameter,
        0,
        &records,
        &["i", "j"],
    )
    .unwrap();
    w.finish().unwrap();
    eprintln!("written {out}");
}
