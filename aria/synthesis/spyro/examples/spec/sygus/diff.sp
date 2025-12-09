//@Description Toy benchmarks to show complex recursive generators.

var {
    int x;
    int y;
    int o;
}

relation {
    f(x, y, o);
}

generator {
    boolean AP -> compare(I, I + I);
    int I -> x | y | o | 0 ;
}

example {
    int -> ??(4) | -1 * ??(4) ;
}

void f(int x, int y, ref int out){
    if (x > y) {
        out = x - y;
    } else {
        out = y - x;
    }
}