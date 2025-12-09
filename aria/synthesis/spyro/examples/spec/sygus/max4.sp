//@Description Toy benchmarks to show complex recursive generators.

var {
    int x1;
    int x2;
    int x3;
    int x4;
    int o;
}

relation {
    max4(x1, x2, x3, x4, o);
}

generator {
    boolean AP -> compare(I, I) ;
    int I -> x1 | x2 | x3 | x4 | o ;
}

example {
    int -> ??(5) | -1 * ??(5) ;
}

void max4(int x1, int x2, int x3, int x4, ref int out){
    if (x1 > x2) {
        if (x1 > x3) {
            if (x1 > x4) {
                out = x1;
            } else {
                out = x4;
            }
        } else {
            if (x3 > x4) {
                out = x3;
            } else {
                out = x4;
            }
        }
    } else {
        if (x2 > x3) {
            if (x2 > x4) {
                out = x2;
            } else {
                out = x4;
            }
        } else {
            if (x3 > x4) {
                out = x3;
            } else {
                out = x4;
            }
        }
    }
}