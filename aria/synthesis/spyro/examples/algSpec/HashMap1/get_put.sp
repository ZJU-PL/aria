//@Description 

var {
    HashTable map;
    Key key;
    Value value;
    HashTable put_out;

    Key key2;
    Value get_out;
}

relation {
    put(map, key, value, put_out);
    get(put_out, key2, get_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | equalKey(K, K) | !equalKey(K, K);
    boolean RHS -> equalValue(get_out, V);
    HashTable M -> map | newHashTable();
    Key K -> key | key2;
    Value V -> null | value | get(M, K);
}

example {
    int -> ??(3) | -1 * ??(3) ;
    Key -> genKey(int);
    Value -> genValue(int);
    HashTable -> newHashTable() | put(HashTable, Key, Value);
    boolean -> true | false;
}

void genKey(int key, ref Key ret) {
    ret = new Key(value = key, hash = key);
}

void genValue(int value, ref Value ret) {
    ret = new Value(value = value);
}