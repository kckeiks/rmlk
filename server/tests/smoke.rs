use rmlk_server::VERSION;

#[test]
fn crate_version_is_present() {
    assert!(!VERSION.is_empty());
}
