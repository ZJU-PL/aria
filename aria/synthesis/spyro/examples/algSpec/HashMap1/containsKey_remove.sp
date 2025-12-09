var {
    HashTable map;
    Key key;
    HashTable remove_out;

    Key key2;
    boolean contains_out;
}

relation {
    remove(map, key, remove_out);
    containsKey(remove_out, key2, contains_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true | equalKey(K, K) | !equalKey(K, K);
    boolean RHS -> contains_out == BB;
    HashTable M -> map | newHashTable();
    Key K -> key | key2;
    boolean BB -> ?? | containsKey(M, K);
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