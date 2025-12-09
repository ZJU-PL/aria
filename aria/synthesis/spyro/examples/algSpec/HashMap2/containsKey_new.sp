var {
    HashTable new_out;

    Key key;
    boolean contains_out;
}

relation {
    newHashTable(new_out);
    containsKey(new_out, key, contains_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true;
    boolean RHS -> contains_out == BB;
    boolean BB -> ??;
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