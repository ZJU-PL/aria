//@Description 

var {
    ArraySet new_out;

    int v;
    boolean contains_out;
}

relation {
    newArraySet(new_out);
    contains(new_out, v, contains_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true;
    boolean RHS -> contains_out == ??;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArraySet -> newArraySet() | add(ArraySet, int);
    boolean -> true | false;
}