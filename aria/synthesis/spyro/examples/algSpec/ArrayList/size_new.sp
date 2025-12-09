//@Description 

var {
    ArrayList new_out;

    int size_out;
}

relation {
    newArrayList(new_out);
    size(new_out, size_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true;
    boolean RHS -> size_out == S;
    int S -> ??(1);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    ArrayList -> newArrayList() | add(ArrayList, int);
    boolean -> true | false;
}