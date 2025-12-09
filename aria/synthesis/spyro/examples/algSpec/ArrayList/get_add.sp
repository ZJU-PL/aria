//@Description 

var {
    ArrayList a;
    int e;
    ArrayList add_out;

    int idx;
    int get_out;
}

relation {
    add(a, e, add_out);
    get(add_out, idx, get_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> get_out == I;
    int S -> size(L) + ??(1) | idx + ??(1) | ??(1);
    ArrayList L -> newArrayList() | a;
    int I -> e | get(L, idx);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArrayList -> newArrayList() | add(ArrayList, int);
    boolean -> true | false;
}