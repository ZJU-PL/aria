//@Description 

var {
    ArrayList new_out;

    int idx;
    E get_out;
}

relation {
    newArrayList(new_out);
    get(new_out, idx, get_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> get_out == I;
    int S -> idx | 0;
    int C -> -1 | 0 | 1;
    E I -> null;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    E -> ??;
    ArrayList -> newArrayList() | add(ArrayList, E);
    boolean -> true | false;
}