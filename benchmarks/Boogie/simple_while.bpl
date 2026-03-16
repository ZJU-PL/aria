implementation SimpleWhile() {
    var x: int;
    x := 10;
    while (x > 0) { x := x - 1; }
    assert x == 0;
}
