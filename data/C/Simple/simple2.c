int main() {
    int x;
    x = 1;
    assume(x >= 1);
    while (x < 5) { x = x + 2; }
    assert(x >= 5);
}