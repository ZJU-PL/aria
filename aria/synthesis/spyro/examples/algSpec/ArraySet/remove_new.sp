//@Description 

var {
    ArraySet add_out;

    int v;
    ArraySet remove_out;
}

relation {
    newArraySet(new_out);
    remove(add_out, v, remove_out);
}

generator {
    boolean AP -> equal(remove_out, AS);
    ArraySet AS -> newArraySet() | add(AS, v);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArraySet -> newArraySet() | add(ArraySet, int);
    boolean -> true | false;
}