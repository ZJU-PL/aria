//@Description 

var {
    HashMap new_out;

    int key;
    boolean containsKey_out;
}

relation {
    newHashMap(new_out);
    containsKey(new_out, key, containsKey_out);
}

generator {
    boolean AP -> containsKey_out == ??;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    HashMap -> newHashMap() | put(HashMap, int, int);
    boolean -> true | false;
}