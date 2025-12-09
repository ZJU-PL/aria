//@Description 

var {
    ArrayList a;
    int e;
    ArrayList add_out;

    int size_out;
}

relation {
    add(a, e, add_out);
    size(add_out, size_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | compare(S, S);
    boolean RHS -> size_out == S;
    int S -> size(L) + ??(1) | ??(1);
    ArrayList L -> newArrayList() | a;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArrayList -> newArrayList() | add(ArrayList, int);
    boolean -> true | false;
}