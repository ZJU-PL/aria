//@Description 

var {
    ArrayList a;
    int minCapacity;
    ArrayList ensure_out;

    int size_out;
}

relation {
    ensureCapacity(a, minCapacity, ensure_out);
    size(ensure_out, size_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> size_out == S + C;
    int S -> minCapacity | size(L) | 0;
    int C -> -1 | 0 | 1;
    ArrayList L -> newArrayList() | a;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    E -> ??;
    ArrayList -> newArrayList() | add(ArrayList, E);
    boolean -> true | false;
}