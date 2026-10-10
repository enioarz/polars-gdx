//! Self-contained round-trip: write a GDX file with the writer, read it back,
//! and assert structure and values. Requires no external data or GAMS install.

use std::sync::Arc;

use gdx::{GdxFile, GdxWriter, Record, SymbolType, ValueField};

fn rec(keys: &[&str], values: [f64; 5]) -> Record {
    Record {
        keys: keys.iter().map(|s| Arc::from(*s)).collect(),
        values,
    }
}

fn write_fixture(path: &std::path::Path) {
    let mut w = GdxWriter::create(path, "gdxcomp-test").unwrap();

    // Set i (dim 1): two elements. Set element "value" is conventionally 0.
    w.write_symbol(
        "i",
        "plants",
        1,
        SymbolType::Set,
        0,
        &[rec(&["seattle"], [0.0; 5]), rec(&["san-diego"], [0.0; 5])],
    )
    .unwrap();

    // Parameter c (dim 2): the value lives in the Level (index 0) slot.
    w.write_symbol(
        "c",
        "transport cost",
        2,
        SymbolType::Parameter,
        0,
        &[
            rec(&["seattle", "new-york"], [0.225, 0.0, 0.0, 0.0, 0.0]),
            rec(&["seattle", "chicago"], [0.153, 0.0, 0.0, 0.0, 0.0]),
            rec(&["san-diego", "topeka"], [0.126, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    .unwrap();

    // Variable x (dim 1): carries all five fields.
    w.write_symbol(
        "x",
        "shipment",
        1,
        SymbolType::Variable,
        0,
        &[rec(&["seattle"], [50.0, 1.5, 0.0, 1e30, 1.0])],
    )
    .unwrap();

    w.finish().unwrap();
}

#[test]
fn write_then_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.gdx");
    write_fixture(&path);

    let file = GdxFile::open(&path).unwrap();

    // Symbol table.
    let names: Vec<&str> = file.symbols().iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["i", "c", "x"]);

    let set_i = file.symbol("i").unwrap();
    assert_eq!(set_i.kind, SymbolType::Set);
    assert_eq!(set_i.dim, 1);
    assert_eq!(set_i.records, 2);

    let par_c = file.symbol("c").unwrap();
    assert_eq!(par_c.kind, SymbolType::Parameter);
    assert_eq!(par_c.dim, 2);
    assert_eq!(par_c.records, 3);
    assert_eq!(par_c.text, "transport cost");

    let var_x = file.symbol("x").unwrap();
    assert_eq!(var_x.kind, SymbolType::Variable);
    assert!(var_x.kind.has_fields());

    // Parameter values (Level slot).
    let c = file.read("c").unwrap();
    assert_eq!(c.len(), 3);
    assert!(c[0]
        .keys
        .iter()
        .zip(["seattle", "new-york"])
        .all(|(a, b)| a.as_ref() == b));
    assert!((c[0].value(ValueField::Level) - 0.225).abs() < 1e-12);
    assert!((c[2].value(ValueField::Level) - 0.126).abs() < 1e-12);

    // Variable fields.
    let x = file.read("x").unwrap();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0].keys[0].as_ref(), "seattle");
    assert!((x[0].value(ValueField::Level) - 50.0).abs() < 1e-12);
    assert!((x[0].value(ValueField::Marginal) - 1.5).abs() < 1e-12);
}

#[test]
fn missing_symbol_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.gdx");
    write_fixture(&path);

    let file = GdxFile::open(&path).unwrap();
    let err = file.read("does-not-exist").unwrap_err();
    assert!(matches!(err, gdx::GdxError::SymbolNotFound(_)));
}

#[test]
fn open_nonexistent_errors() {
    let err = GdxFile::open("/nonexistent/path/to/file.gdx")
        .err()
        .expect("opening a missing file should fail");
    assert!(matches!(err, gdx::GdxError::OpenRead { .. }));
}

#[test]
fn raw_limits_are_exact_and_count_only_matches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("limits.gdx");
    write_fixture(&path);
    let file = GdxFile::open(&path).unwrap();
    let info = file.symbol("c").unwrap();
    let full = file
        .read_symbol_raw(info, ValueField::Level, None, None)
        .unwrap();
    let skip_first = |keys: &[i32]| {
        if keys[1] == full.keys[1][0] as i32 + 1 {
            gdx::RecordAction::Skip
        } else {
            gdx::RecordAction::Accept
        }
    };
    for pred in [
        None,
        Some(&skip_first as &dyn Fn(&[i32]) -> gdx::RecordAction),
    ] {
        let expected = file
            .read_symbol_raw(info, ValueField::Level, pred, None)
            .unwrap();
        for limit in [0, 1, 2, 10] {
            let actual = file
                .read_symbol_raw(info, ValueField::Level, pred, Some(limit))
                .unwrap();
            let n = limit.min(expected.len());
            assert_eq!(actual.values, expected.values[..n]);
            for (actual, expected) in actual.keys.iter().zip(&expected.keys) {
                assert_eq!(actual, &expected[..n]);
            }
        }
    }
}

#[test]
fn invalid_writer_shapes_leave_writer_usable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid-shapes.gdx");
    let mut writer = GdxWriter::create(&path, "validation-test").unwrap();
    let valid = rec(&["a", "b"], [1.0, 0.0, 0.0, 0.0, 0.0]);

    for keys in [&["a"][..], &["a", "b", "c"][..]] {
        // A bad later record must be rejected before even the valid one is written.
        let err = writer
            .write_symbol(
                "p",
                "",
                2,
                SymbolType::Parameter,
                0,
                &[valid.clone(), rec(keys, [1.0; 5])],
            )
            .unwrap_err();
        assert!(matches!(err, gdx::GdxError::InvalidWriteInput(_)));
    }
    for domains in [&["i"][..], &["i", "j", "k"][..]] {
        let err = writer
            .write_symbol_with_domains(
                "p",
                "",
                2,
                SymbolType::Parameter,
                0,
                std::slice::from_ref(&valid),
                domains,
            )
            .unwrap_err();
        assert!(matches!(err, gdx::GdxError::InvalidWriteInput(_)));
    }
    for dim in [gdx_sys::GMS_MAX_INDEX_DIM + 1, usize::MAX] {
        let err = writer
            .write_symbol("p", "", dim, SymbolType::Parameter, 0, &[])
            .unwrap_err();
        assert!(matches!(err, gdx::GdxError::InvalidWriteInput(_)));
    }
    writer
        .write_symbol_with_domains(
            "p",
            "",
            2,
            SymbolType::Parameter,
            0,
            std::slice::from_ref(&valid),
            &["i", "j"],
        )
        .unwrap();
    writer.finish().unwrap();

    let file = GdxFile::open(&path).unwrap();
    assert_eq!(file.symbols().len(), 1);
    assert_eq!(file.symbol("p").unwrap().domains, ["i", "j"]);
    let records = file.read("p").unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].keys, valid.keys);
    assert_eq!(records[0].values[0], 1.0);
}
