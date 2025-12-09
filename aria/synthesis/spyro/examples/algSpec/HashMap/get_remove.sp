//@Description 

var {
    HashMap map;
    int key;
    HashMap remove_out;

    int key2;
    int get_out;
    boolean err;

    boolean err_sub;
}

relation {
    remove(map, key, remove_out);
    get(remove_out, key2, err, get_out);
}

generator {
    boolean AP -> !GUARD || !GUARD || RHS;
    boolean GUARD -> true | K == K | K != K | containsKey(map, K) | !containsKey(map, K);
    boolean RHS -> err | get_out == V && !err;
    HashMap M -> map | newHashMap();
    int K -> key | key2;
    int V -> get(M, K, err_sub);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    HashMap -> newHashMap() | put(HashMap, int, int);
    boolean -> true | false;
}