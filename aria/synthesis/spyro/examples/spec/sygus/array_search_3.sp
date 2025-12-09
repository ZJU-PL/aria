//@Description array_search_3.sl of SyGuS array track.
//recommend option --num-atoms-max 5

var {
    int x1;
    int x2;
    int x3;
    int k;
    int o;
}

relation {
    f(x1, x2, x3, k, o);
}

generator {
    boolean AP -> compare(I, I) | compare(S, S);
    int S -> 0 | 1 | 2 | 3 | o;
    int I -> x1 | x2 | x3 | k;
}

example {
    int -> ??(4) | -1 * ??(4) ;
}

void f(int x1, int x2, int x3, int k, ref int out){
    if (k < x1) {
        out = 0;
    } else if (k < x2) {
        out = 1;
    } else if (k < x3) {
        out = 2;
    } else {
        out = 3;
    }
}