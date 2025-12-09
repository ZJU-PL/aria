//@Description 

var {
    ArrayList a;
    int idx;
    E e;
    ArrayList set_out;

    int size_out;
}

relation {
    set(a, idx, e, set_out);
    size(set_out, size_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> size_out == (S + C);
    int S -> size(L) | idx | 0;
    int C -> -1 | 0 | 1;
    ArrayList L -> newArrayList() | a;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    E -> ??;
    ArrayList -> newArrayList() | add(ArrayList, E);
    boolean -> true | false;
}