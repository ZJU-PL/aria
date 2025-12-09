//@Description 

var {
    HashMap map;
    int key;
    int value;
    HashMap put_out;

    int key2;
    boolean containsKey_out;
}

relation {
    put(map, key, value, put_out);
    containsKey(put_out, key2, containsKey_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | K == K | K != K;
    boolean RHS -> containsKey_out == BB;
    HashMap M -> map | newHashMap();
    int K -> key | key2;
    boolean BB -> ?? | containsKey(M, K);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    HashMap -> newHashMap() | put(HashMap, int, int);
    boolean -> true | false;
}