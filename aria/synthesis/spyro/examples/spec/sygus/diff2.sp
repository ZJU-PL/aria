//@Description Toy benchmarks to show complex recursive generators.

var {
    int x1;
    int y1;
    int o1;

    int x2;
    int y2;
    int o2;
}

relation {
    f(x1, y1, o1);
    f(x2, y2, o2);
}

generator {
    boolean AP -> I == I | I != I;
    int I -> x1 | x2 | y1 | y2 | o1 | o2;
}

example {
    int -> ??(5) | -1 * ??(5) ;
}

void f(int x, int y, ref int out){
    if (x > y) {
        out = x - y;
    } else {
        out = y - x;
    }
}