struct Key {
    int value;
    int hash;
}

void equalKey(Key k1, Key k2, ref boolean ret) {
    if (k1 == null || k2 == null) {
        ret = (k1 == k2);
    } else {
        ret = (k1.value == k2.value) && (k1.hash == k2.hash);
    }
}

struct Value {
    int value;
}

void equalValue(Value v1, Value v2, ref boolean ret) {
    if (v1 == null || v2 == null) {
        ret = (v1 == v2);
    } else {
        ret = (v1.value == v2.value);
    }    
}

struct KVPair {
    Key key;
    Value value;
}

struct HashTable {
    int size;
    int mod;
    int numberOfSlots;
    KVPair[size] bucketHash;
    ArrayList overflow;
    int[mod] sizeBucket;
    int numberOfElements;
}

void newHashTable(ref HashTable ret) {
    ret = new HashTable(
        size = 5, mod = 2, 
        numberOfSlots = 2, numberOfElements = 0
    );

    list_empty(ret.overflow);
}

void table_copy(HashTable tb, ref HashTable ret) {
    ret = new HashTable(
        size = tb.size, mod = tb.mod,
        numberOfSlots = tb.numberOfSlots, numberOfElements = tb.numberOfElements
    );

    list_copy(tb.overflow, ret.overflow);
    ret.bucketHash[0:tb.size] = tb.bucketHash[0:tb.size];
    ret.sizeBucket[0:tb.mod] = tb.sizeBucket[0:tb.mod];
}

void get(HashTable tb, Key key, ref Value ret) {
    if (key == null) {
        ret = null;
        return;
    }

    int integerKey = key.hash % tb.mod;
    if (integerKey < 0) { integerKey = -1 * integerKey; }

    int index = tb.numberOfSlots * integerKey;
    for (int i = index; i < index + tb.sizeBucket[integerKey]; i++) {
        KVPair p = tb.bucketHash[i];
        boolean b;
        equalKey(key, p.key, b);
        if (b) {
            ret = p.value;
            return;
        }
    }

    if (tb.sizeBucket[integerKey] == tb.numberOfSlots) {
        ArrayList os = tb.overflow;
        int sz;
        list_size(os, sz);
        for (int i = 0; i < sz; i++) {
            KVPair p;
            boolean b;
            list_get(os, i, p);
            equalKey(key, p.key, b);
            if (b) {
                ret = p.value;
                return;
            }
        }
    }

    ret = null;
}

void put(HashTable tb, Key key, Value value, ref HashTable ret) {
    remove(tb, key, ret);
    int integerKey = key.hash % tb.mod;
    if (integerKey < 0) { integerKey = -1 * integerKey; }

    // check if there is a place in bucketin array or not
    if (ret.sizeBucket[integerKey] != ret.numberOfSlots) {
        int index = ret.numberOfSlots * integerKey + ret.sizeBucket[integerKey];
        KVPair p = new KVPair(key=key, value=value);
        ret.bucketHash[index] = p;
        ret.sizeBucket[integerKey] = ret.sizeBucket[integerKey] + 1;
    } else {
        ArrayList l_tmp;
        KVPair p = new KVPair(key=key, value=value);
        list_add(ret.overflow, p, l_tmp);
        ret.overflow = l_tmp;
    }

    ret.numberOfElements = ret.numberOfElements + 1;
    if (ret.numberOfElements * 4 > ret.size * 3) {
        HashTable tb_tmp;
        rehash(ret, tb_tmp);
        ret = tb_tmp;
    }
}

void put_no_rehash(HashTable tb, Key key, Value value, ref HashTable ret) {
    remove(tb, key, ret);
    int integerKey = key.hash % tb.mod;
    if (integerKey < 0) { integerKey = -1 * integerKey; }

    // check if there is a place in bucketin array or not
    if (ret.sizeBucket[integerKey] != ret.numberOfSlots) {
        int index = ret.numberOfSlots * integerKey + ret.sizeBucket[integerKey];
        KVPair p = new KVPair(key=key, value=value);
        ret.bucketHash[index] = p;
        ret.sizeBucket[integerKey] = ret.sizeBucket[integerKey] + 1;
    } else {
        ArrayList l_tmp;
        KVPair p = new KVPair(key=key, value=value);
        list_add(ret.overflow, p, l_tmp);
        ret.overflow = l_tmp;
    }

    ret.numberOfElements = ret.numberOfElements + 1;
}

void rehash(HashTable tb, ref HashTable ret) {
    ArrayList tmp_l1;
    ArrayList tmp_l2;
    list_empty(tmp_l1);

    for (int i = 0; i < tb.mod; i++) {
        for (int j = 0; j < tb.sizeBucket[i]; j++) {
            int index = i * tb.numberOfSlots + j;
            list_add(tmp_l1, tb.bucketHash[index], tmp_l2);
            tmp_l1 = tmp_l2;
        }
    }

    int sz;
    list_size(tb.overflow, sz);
    for (int i = 0; i < sz; i++) {
        KVPair p;
        list_get(tb.overflow, i, p);
        list_add(tmp_l1, p, tmp_l2);
        tmp_l1 = tmp_l2;
    }

    ret = new HashTable(
        size = tb.size * 4, mod = tb.mod * 2,
        numberOfSlots = tb.numberOfSlots * 2, numberOfElements = 0
    );
    list_empty(ret.overflow);

    int sz2;
    list_size(tmp_l1, sz2);
    for (int i = 0; i < sz2; i++) {
        KVPair p;
        HashTable tmp_tb;
        list_get(tmp_l1, i, p);
        put_no_rehash(ret, p.key, p.value, tmp_tb);
        ret = tmp_tb;
    }
}

void remove(HashTable tb, Key key, ref HashTable ret) {
    int integerKey = key.hash % tb.mod;
    if (integerKey < 0) { integerKey = -1 * integerKey; }

    int index = tb.numberOfSlots * integerKey;
    boolean flag = false;
    int add = index + tb.sizeBucket[integerKey];

    table_copy(tb, ret);

    for (int i = index; i < add; i++) {
        KVPair tmp = ret.bucketHash[i];
        boolean b;
        equalKey(tmp.key, key, b);
        if (b) { flag = true; } 
        else if (flag) { ret.bucketHash[i-1] = tmp; }
    }

    if (flag) {
        ret.numberOfElements = ret.numberOfElements - 1;
        ret.sizeBucket[integerKey] = ret.sizeBucket[integerKey] - 1;
    }

    else if (ret.sizeBucket[integerKey] == ret.numberOfSlots) {
        int sz;
        list_size(ret.overflow, sz);

        for (int i = 0; i < sz; i++) {
            KVPair tmp;
            boolean b;

            list_get(ret.overflow, i, tmp);
            equalKey(key, tmp.key, b);

            if (b) {
                ArrayList tmp_list;
                list_remove(ret.overflow, i, tmp_list);
                ret.overflow = tmp_list;
                ret.numberOfElements = ret.numberOfElements - 1;
                return;
            }
        }
    }
}

void containsKey(HashTable tb, Key key, ref boolean ret) {
    Value check;
    get(tb, key, check);
    ret = (check != null);
}

void isEmpty(HashTable tb, ref boolean ret) {
    ret = (tb.numberOfElements == 0);
}

void size(HashTable tb, ref int ret) {
    ret = tb.numberOfElements;
}

