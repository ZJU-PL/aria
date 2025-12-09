//@Description 

var {
    HashMap new_out;

    int key;
    int get_out;
    boolean err;
}

relation {
    newHashMap(new_out);
    get(new_out, key, err, get_out);
}

generator {
    boolean AP -> err;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    HashMap -> newHashMap() | put(HashMap, int, int);
    boolean -> true | false;
}