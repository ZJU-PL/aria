//@Description Toy benchmarks to show complex recursive generators.

var {
    int x;
    int o;
}

relation {
    abs(x, o);
}

generator {
    boolean AP -> compare(C * x + C * o + C, 0) ;
    int C -> ??(3) - 3 ;
}

example {
    int -> ?? | -1 * ?? ;
}

void abs(int x, ref int out){
    if (x < 0) {
        out = -x;
    } else {
        out = x;
    }
}