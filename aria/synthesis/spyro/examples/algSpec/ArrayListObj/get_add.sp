//@Description 

var {
    ArrayList a;
    E e;
    ArrayList add_out;

    int idx;
    E get_out;
}

relation {
    add(a, e, add_out);
    get(add_out, idx, get_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> get_out == I;
    int S -> size(L) | idx | 0;
    int C -> -1 | 0 | 1;
    ArrayList L -> newArrayList() | a;
    E I -> null | e | get(L, S + C);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    E -> ??;
    ArrayList -> newArrayList() | add(ArrayList, E);
    boolean -> true | false;
}