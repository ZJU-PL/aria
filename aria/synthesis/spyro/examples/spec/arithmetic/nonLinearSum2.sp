//@Description Toy benchmarks to show complex recursive generators.

var {
    int x;
    int o;
}

relation {
    sum(x, o);
}

generator {
    boolean AP -> compare(C * x + C * o + C * x * x + C, 0) ;
    int C -> ??(3) - 4  ;
}

example {
    int -> ??(5) | -1 * ??(5) ;
}

void sum(int x, ref int out){
    if (x > 0) {
        int sub_out;
        sum(x-1, sub_out);
        out = sub_out + x;
    } else {
        out = 0;
    }
}