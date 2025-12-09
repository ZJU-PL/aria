//@Description 

var {
    ArraySet arr;
    int v1;
    ArraySet add_out;

    int v2;
    ArraySet remove_out;
}

relation {
    add(arr, v1, add_out);
    remove(add_out, v2, remove_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | I == I | I != I;
    boolean RHS -> equal(remove_out, AS);
    int I -> v1 | v2;
    ArraySet AS -> newArraySet() | arr | add(AS, I) | remove(arr, I);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArraySet -> newArraySet() | add(ArraySet, int);
    boolean -> true | false;
}