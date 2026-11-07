//! Generate the frozen M7 semantic fixture (docs/m7-fixture-expectations.md)
//! with the pinned gdxrs core writer. Usage:
//!
//! ```sh
//! cargo run -p polars-gdx --release --example gen_m7_fixtures -- [output.gdx]
//! ```
//!
//! The fixture is the independently authored input recipe for task 042:
//! UELs registered as aa, bb, cc, dd, AA, seattle, Seattle (IDs 1-7 in that
//! order), acronym ID 7 ("acseven"), set I over IDs 1 and 3, alias J -> I,
//! empty parameter D with strict domain J, typed-values parameter P over
//! IDs 1-7 (EPS, UNDEF, NA, +INF, -INF, 1.25, acronym 7), case-sensitivity
//! parameter C (aa -> 10, AA -> 20), scalars S = 1.25 and Z (no stored
//! records), and one variable/equation record each. Expected results are
//! fixed in docs/m7-fixture-expectations.md and asserted by
//! tests/test_m7_task042.py; never regenerate expectations from output.

use std::path::Path;

use gdxrs::{
    AcronymId, MetadataOptions, SpecialValue, SymbolReference, SymbolType, Value, WriteDomain,
    WriteOptions, WriteSymbol, Writer,
};

fn main() {
    let mut args = std::env::args_os();
    let _ = args.next();
    let output = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new("tests/data/m7/m7-semantics.gdx").to_path_buf());
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }
    let scratch = std::env::temp_dir().join("gen-m7-fixtures");
    std::fs::create_dir_all(&scratch).expect("create scratch directory");
    let mut writer = Writer::create(&output, WriteOptions::default()).expect("create writer");
    writer
        .begin_metadata(&MetadataOptions::new(&scratch))
        .expect("begin metadata");

    let aa = writer.register_uel(b"aa").expect("register aa");
    let bb = writer.register_uel(b"bb").expect("register bb");
    let cc = writer.register_uel(b"cc").expect("register cc");
    let dd = writer.register_uel(b"dd").expect("register dd");
    let upper_aa = writer.register_uel(b"AA").expect("register AA");
    let seattle = writer.register_uel(b"seattle").expect("register seattle");
    let _ = seattle;
    let upper_seattle = writer.register_uel(b"Seattle").expect("register Seattle");
    let _ = upper_seattle;
    assert_eq!(aa.get(), 1);
    assert_eq!(bb.get(), 2);
    assert_eq!(cc.get(), 3);
    assert_eq!(dd.get(), 4);
    assert_eq!(upper_aa.get(), 5);

    writer
        .add_acronym(
            AcronymId::new(7).expect("acronym id"),
            b"acseven",
            b"independently specified acronym",
        )
        .expect("add acronym");

    let set_i = writer
        .register_symbol(&WriteSymbol::new(b"I", SymbolType::Set, 1))
        .expect("register set I");
    let mut spec = WriteSymbol::new(b"J", SymbolType::Alias, 1);
    spec.alias_target = Some(SymbolReference::Registered(set_i));
    let alias_j = writer.register_symbol(&spec).expect("register alias J");
    let param_d = writer
        .register_symbol(&WriteSymbol::new(b"D", SymbolType::Parameter, 1))
        .expect("register parameter D");
    writer
        .set_domains(
            param_d,
            &[WriteDomain::Strict(SymbolReference::Registered(alias_j))],
        )
        .expect("set D domains");
    let param_p = writer
        .register_symbol(&WriteSymbol::new(b"P", SymbolType::Parameter, 1))
        .expect("register parameter P");
    let param_c = writer
        .register_symbol(&WriteSymbol::new(b"C", SymbolType::Parameter, 1))
        .expect("register parameter C");
    let scalar_s = writer
        .register_symbol(&WriteSymbol::new(b"S", SymbolType::Parameter, 0))
        .expect("register scalar S");
    let scalar_z = writer
        .register_symbol(&WriteSymbol::new(b"Z", SymbolType::Parameter, 0))
        .expect("register scalar Z");
    let var_v = writer
        .register_symbol(&WriteSymbol::new(b"V", SymbolType::Variable, 1))
        .expect("register variable V");
    let eq_q = writer
        .register_symbol(&WriteSymbol::new(b"Q", SymbolType::Equation, 1))
        .expect("register equation Q");

    // Set records store a zero-based set-text ID in the value slot
    // (0 = no text); the raw read projects it as 0.0.
    let mut rows = writer.begin_symbol(set_i, None).expect("write set I");
    rows.write(&[aa], &[Value::ZERO]).expect("set record");
    rows.write(&[cc], &[Value::ZERO]).expect("set record");
    rows.finish().expect("finish set I");

    let rows = writer.begin_symbol(param_d, None).expect("write D");
    rows.finish().expect("finish D");

    let mut rows = writer.begin_symbol(param_p, None).expect("write P");
    let acronym = AcronymId::new(7)
        .expect("acronym id")
        .value()
        .expect("acronym value");
    for (id, value) in [
        (aa, Value::Special(SpecialValue::Epsilon)),
        (bb, Value::Special(SpecialValue::Undefined)),
        (cc, Value::Special(SpecialValue::NotAvailable)),
        (dd, Value::Special(SpecialValue::PositiveInfinity)),
        (upper_aa, Value::Special(SpecialValue::NegativeInfinity)),
        (seattle, Value::NumberBits(1.25f64.to_bits())),
        (upper_seattle, acronym),
    ] {
        rows.write(&[id], &[value]).expect("typed record");
    }
    rows.finish().expect("finish P");

    let mut rows = writer.begin_symbol(param_c, None).expect("write C");
    rows.write(&[aa], &[Value::NumberBits(10.0f64.to_bits())])
        .expect("C record");
    rows.write(&[upper_aa], &[Value::NumberBits(20.0f64.to_bits())])
        .expect("C record");
    rows.finish().expect("finish C");

    let mut rows = writer.begin_symbol(scalar_s, None).expect("write S");
    rows.write(&[], &[Value::NumberBits(1.25f64.to_bits())])
        .expect("S record");
    rows.finish().expect("finish S");

    let rows = writer.begin_symbol(scalar_z, None).expect("write Z");
    rows.finish().expect("finish Z");

    let mut rows = writer.begin_symbol(var_v, None).expect("write V");
    rows.write(
        &[aa],
        &[
            Value::NumberBits(1.5f64.to_bits()),
            Value::NumberBits(2.5f64.to_bits()),
            Value::NumberBits(0.5f64.to_bits()),
            Value::NumberBits(3.5f64.to_bits()),
            Value::NumberBits(1.0f64.to_bits()),
        ],
    )
    .expect("V record");
    rows.finish().expect("finish V");

    let mut rows = writer.begin_symbol(eq_q, None).expect("write Q");
    rows.write(
        &[bb],
        &[
            Value::NumberBits(4.5f64.to_bits()),
            Value::NumberBits(0.0f64.to_bits()),
            Value::NumberBits(0.0f64.to_bits()),
            Value::NumberBits(10.0f64.to_bits()),
            Value::NumberBits(1.0f64.to_bits()),
        ],
    )
    .expect("Q record");
    rows.finish().expect("finish Q");

    let stats = writer.finish().expect("finish writer");
    drop(writer);
    let _ = std::fs::remove_dir_all(&scratch);
    println!(
        "wrote {} ({} bytes, status {:?})",
        output.display(),
        stats.output_bytes,
        stats.status
    );
}
