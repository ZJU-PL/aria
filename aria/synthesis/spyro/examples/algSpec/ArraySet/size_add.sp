//@Description 

var {
    ArraySet a;
    int e;
    ArraySet add_out;

    int size_out;
}

relation {
    add(a, e, add_out);
    size(add_out, size_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | compare(S, S) 
                    | contains(a, e) | !contains(a, e);
    boolean RHS -> size_out == S;
    int S -> size(L) + ??(1) | ??(1);
    ArraySet L -> newArraySet() | a;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArraySet -> newArraySet() | add(ArraySet, int);
    boolean -> true | false;
}