//@Description Toy benchmarks to show complex recursive generators.

var {
    int x;
    int o;
}

relation {
    sum(x, o);
}

generator {
    boolean B -> compare(I, I) ;
    int I -> x | o | -x | -o | 0 | 1;
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