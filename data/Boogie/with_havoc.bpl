implementation WithHavoc() {
    var x, random: int;
    x := 0;
    while (x < 10) {
        havoc random; assume random >= 0 && random <= 1;
        x := x + random;
    }
    assert x >= 10;
}
