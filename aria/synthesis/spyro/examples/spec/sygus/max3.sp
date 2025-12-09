//@Description Toy benchmarks to show complex recursive generators.

var {
    int x1;
    int x2;
    int x3;
    int o;
}

relation {
    max3(x1, x2, x3, o);
}

generator {
    boolean AP -> compare(I, I) ;
    int I -> x1 | x2 | x3 | o ;
}

example {
    int -> ??(5) | -1 * ??(5) ;
}

void max3(int x1, int x2, int x3, ref int out){
    if (x1 > x2) {
        if (x1 > x3) {
            out = x1;
        } else {
            out = x3;
        }
    } else {
        if (x2 > x3) {
            out = x2;
        } else {
            out = x3;
        }
    }
}