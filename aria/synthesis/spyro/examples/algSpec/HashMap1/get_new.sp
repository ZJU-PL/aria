var {
    HashTable new_out;

    Key key;
    Value get_out;
}

relation {
    newHashTable(new_out);
    get(new_out, key, get_out);
}

generator {
    boolean AP -> !GUARD || RHS;
    boolean GUARD -> true;
    boolean RHS -> equalValue(get_out, V);
    Value V -> null;
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