//@Description array_search_2.sl of SyGuS array track.

var {
    int x1;
    int x2;
    int k;
    int o;
}

relation {
    f(x1, x2, k, o);
}

generator {
    boolean AP -> compare(I, I) | compare(S, S);
    int S -> 0 | 1 | 2 | o;
    int I -> x1 | x2 | k;
}

example {
    int -> ??(4) | -1 * ??(4) ;
}

void f(int x1, int x2, int k, ref int out){
    if (k < x1) {
        out = 0;
    } else if (k < x2) {
        out = 1;
    } else {
        out = 2;
    }
}