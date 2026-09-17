use fixture_rust::Counter;

#[test]
fn test_counter() {
    let mut c = Counter::new();
    c.increment();
    assert_eq!(c.count, 1);
}
