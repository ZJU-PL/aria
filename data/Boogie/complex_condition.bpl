implementation ComplexWhile() {
    var a, b, counter: int;
    a := 10; b := 5; counter := 0;
    while (a > 0 && b > 0) {
        a := a - 1; b := b - 1; counter := counter + 1;
    }
    assert counter <= 10;
}
