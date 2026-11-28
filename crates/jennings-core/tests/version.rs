use std::fs;

#[test]
fn crate_version_is_the_python_project_version() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyproject.toml");
    let text = fs::read_to_string(path).unwrap();
    let project = text.split("[project]").nth(1).unwrap();
    let version = project
        .lines()
        .take_while(|l| !l.starts_with('['))
        .find_map(|l| l.strip_prefix("version = "))
        .unwrap()
        .trim()
        .trim_matches('"');
    assert_eq!(
        version,
        env!("CARGO_PKG_VERSION"),
        "pyproject.toml against Cargo.toml"
    );
}
