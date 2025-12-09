//@Description 

var {
    ArrayList a;
    int idx1;
    ArrayList remove_out;

    int idx2;
    E get_out;
}

relation {
    remove(a, idx1, remove_out);
    get(remove_out, idx2, get_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> get_out == I;
    int S -> size(L) | idx1 | idx2 | 0;
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