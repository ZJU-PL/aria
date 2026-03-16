implementation MultipleVars() {
    var x, y: int;
    x := 5; y := 0;
    while (x > 0) { x := x - 1; y := y + 1; }
    assert x + y == 5;
}
