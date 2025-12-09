//@Description 

var {
    HashMap map;
    int key;
    int value;
    HashMap put_out;

    int key2;
    int value2;
    HashMap put_out2;
}

relation {
    put(map, key, value, put_out);
    put(put_out, key2, value2, put_out2);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | K == K | K != K | V == V | V != V;
    boolean RHS -> equalMap(put_out2, M);
    HashMap M -> map | newHashMap() | put(map, K, V);
    int K -> key | key2;
    int V -> value | value2;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    HashMap -> newHashMap() | put(HashMap, int, int);
    boolean -> true | false;
}