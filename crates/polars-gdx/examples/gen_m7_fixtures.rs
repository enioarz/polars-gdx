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
    let default = Path::new("tests/data/m7").to_path_buf();
    match args
        .next()
        .as_deref()
        .map(|a| a.to_string_lossy().to_string())
    {
        None => {
            let semantics = default.join("m7-semantics.gdx");
            fresh(&semantics);
            write_semantics(&semantics);
            let dictionary = default.join("m7-dictionary.gdx");
            fresh(&dictionary);
            write_dictionary(&dictionary);
            let labelbytes = default.join("m7-labelbytes.gdx");
            fresh(&labelbytes);
            write_labelbytes(&labelbytes);
            let growing = default.join("m7-labels20k.gdx");
            fresh(&growing);
            write_growing(&growing);
        }
        Some(name) => match name.as_str() {
            "m7-semantics.gdx" => {
                let path = default.join(&name);
                fresh(&path);
                write_semantics(&path)
            }
            "m7-dictionary.gdx" => {
                let path = default.join(&name);
                fresh(&path);
                write_dictionary(&path)
            }
            "m7-labelbytes.gdx" => {
                let path = default.join(&name);
                fresh(&path);
                write_labelbytes(&path)
            }
            "m7-labels20k.gdx" => {
                let path = default.join(&name);
                fresh(&path);
                write_growing(&path)
            }
            other => panic!("unknown fixture {other:?}"),
        },
    }
}

fn fresh(output: &Path) {
    if output.exists() {
        std::fs::remove_file(output).expect("remove existing fixture");
    }
}

fn scratch_dir() -> std::path::PathBuf {
    let scratch = std::env::temp_dir().join("gen-m7-fixtures");
    std::fs::create_dir_all(&scratch).expect("create scratch directory");
    scratch
}

/// Dictionary-identity fixture (docs/m7-fixture-expectations.md): UELs
/// registered in order aa, bb, cc, dd (IDs 1-4); parameter P with ordered
/// records (1,10), (2,20), (3,30), (4,40); parameter ZEROS with +0.0 and
/// -0.0 records for the adapter conversion expectation (the writer's
/// Preserve policy keeps the signed-zero bits on disk).
fn write_dictionary(output: &Path) {
    let scratch = scratch_dir();
    let mut writer = Writer::create(output, WriteOptions::default()).expect("create writer");
    writer
        .begin_metadata(&MetadataOptions::new(&scratch))
        .expect("begin metadata");
    let aa = writer.register_uel(b"aa").expect("register aa");
    let bb = writer.register_uel(b"bb").expect("register bb");
    let cc = writer.register_uel(b"cc").expect("register cc");
    let dd = writer.register_uel(b"dd").expect("register dd");
    assert_eq!((aa.get(), bb.get(), cc.get(), dd.get()), (1, 2, 3, 4));
    let p = writer
        .register_symbol(&WriteSymbol::new(b"P", SymbolType::Parameter, 1))
        .expect("register P");
    let zeros = writer
        .register_symbol(&WriteSymbol::new(b"ZEROS", SymbolType::Parameter, 1))
        .expect("register ZEROS");
    let mut rows = writer.begin_symbol(p, None).expect("write P");
    for (id, value) in [(aa, 10.0f64), (bb, 20.0), (cc, 30.0), (dd, 40.0)] {
        rows.write(&[id], &[Value::NumberBits(value.to_bits())])
            .expect("P record");
    }
    rows.finish().expect("finish P");
    let mut rows = writer.begin_symbol(zeros, None).expect("write ZEROS");
    rows.write(&[aa], &[Value::NumberBits(0.0f64.to_bits())])
        .expect("ZEROS record");
    rows.write(&[bb], &[Value::NumberBits((-0.0f64).to_bits())])
        .expect("ZEROS record");
    rows.finish().expect("finish ZEROS");
    let stats = writer.finish().expect("finish writer");
    println!("wrote {} ({} bytes)", output.display(), stats.output_bytes);
}

/// Label-bytes fixture: raw label bytes `61 61` and `ff 62` as IDs 1 and 2
/// with values 10 and 20. `ff 62` is not valid UTF-8: mode A must reject it
/// before output while the bounded ID mode never decodes labels.
fn write_labelbytes(output: &Path) {
    let scratch = scratch_dir();
    let mut writer = Writer::create(output, WriteOptions::default()).expect("create writer");
    writer
        .begin_metadata(&MetadataOptions::new(&scratch))
        .expect("begin metadata");
    let aa = writer.register_uel(b"\x61\x61").expect("register aa bytes");
    let raw = writer
        .register_uel(b"\xff\x62")
        .expect("register raw bytes");
    assert_eq!((aa.get(), raw.get()), (1, 2));
    let b = writer
        .register_symbol(&WriteSymbol::new(b"B", SymbolType::Parameter, 1))
        .expect("register B");
    let mut rows = writer.begin_symbol(b, None).expect("write B");
    rows.write(&[aa], &[Value::NumberBits(10.0f64.to_bits())])
        .expect("B record");
    rows.write(&[raw], &[Value::NumberBits(20.0f64.to_bits())])
        .expect("B record");
    rows.finish().expect("finish B");
    let stats = writer.finish().expect("finish writer");
    println!("wrote {} ({} bytes)", output.display(), stats.output_bytes);
}

/// Growing-table fixture: 20,000 six-byte labels and a 20,000-record
/// parameter over them, for bounded-mode reads whose dictionary is far
/// larger than a tiny admission would allow.
fn write_growing(output: &Path) {
    const COUNT: u64 = 20_000;
    let scratch = scratch_dir();
    let mut writer = Writer::create(output, WriteOptions::default()).expect("create writer");
    writer
        .begin_metadata(&MetadataOptions::new(&scratch))
        .expect("begin metadata");
    let mut ids: Vec<gdxrs::WriterUelId> = Vec::with_capacity(COUNT as usize);
    writer
        .register_uels_fresh(
            (0..COUNT).map(|i| Ok::<Vec<u8>, gdxrs::Error>(format!("u{i:05}").into_bytes())),
            |_ordinal, id| {
                ids.push(id);
                Ok(())
            },
        )
        .expect("bulk register labels");
    let big = writer
        .register_symbol(&WriteSymbol::new(b"BIG", SymbolType::Parameter, 1))
        .expect("register BIG");
    let mut rows = writer.begin_symbol(big, None).expect("write BIG");
    for (i, id) in ids.iter().enumerate() {
        rows.write(
            std::slice::from_ref(id),
            &[Value::NumberBits(((i % 1000) as f64).to_bits())],
        )
        .expect("BIG record");
    }
    rows.finish().expect("finish BIG");
    let stats = writer.finish().expect("finish writer");
    println!(
        "wrote {} ({} bytes, {COUNT} labels)",
        output.display(),
        stats.output_bytes
    );
}

fn write_semantics(output: &Path) {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }
    let scratch = scratch_dir();
    let mut writer = Writer::create(output, WriteOptions::default()).expect("create writer");
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
