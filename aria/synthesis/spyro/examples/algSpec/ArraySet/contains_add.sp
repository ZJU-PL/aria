//@Description 

var {
    ArraySet arr;
    int v1;
    ArraySet add_out;

    int v2;
    boolean contains_out;
}

relation {
    add(arr, v1, add_out);
    contains(add_out, v2, contains_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | I == I | I != I;
    boolean RHS -> contains_out == B;
    int I -> v1 | v2;
    ArraySet AS -> newArraySet() | arr;
    boolean B -> contains(AS, I) | ??;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArraySet -> newArraySet() | add(ArraySet, int);
    boolean -> true | false;
}