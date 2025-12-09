//@Description 

var {
    ArrayList a;
    int minCapacity;
    ArrayList ensure_out;

    int idx;
    E get_out;
}

relation {
    ensureCapacity(a, minCapacity, ensure_out);
    get(ensure_out, idx, get_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> get_out == I;
    int S -> size(L) | minCapacity | idx | 0;
    int C -> -1 | 0 | 1;
    ArrayList L -> newArrayList() | a;
    E I -> null | get(L, S + C);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    E -> ??;
    ArrayList -> newArrayList() | add(ArrayList, E);
    boolean -> true | false;
}