#define MARGIN 5

struct Node {
    int key;
    int value;
    int hash;
}

struct HashMap {
    int size;
    Node[size + MARGIN] elementData;
}

// From Integer.java of JLibSketch
void hashCode(int key, ref int ret) {
    ret = key;
}

void equalNode(Node n1, Node n2, ref boolean ret) {
    if (n1 == null || n2 == null) {
        ret = n1 == n2;
    } else {
        ret = n1.key == n2.key 
            && n1.value == n2.value 
            && n1.hash == n2.hash;
    }
}

void equalMap(HashMap m1, HashMap m2, ref boolean ret) {
    assert m1 != null;
    assert m2 != null;

    if (m1.size != m2.size) {
        ret = false;
        return;
    }

    int i;
    for (i=0; i < m1.size + MARGIN; i++) {
        boolean eq;
        equalNode(m1.elementData[i], m2.elementData[i], eq);
        if (!eq) {
            ret = false;
            return;
        }
    }
    ret = true;
    return;
}

void newHashMap(ref HashMap ret) {
    Node[MARGIN] elementData;

    ret = new HashMap(size=0, elementData=elementData);
}

void size(HashMap map, ref int ret) {
    ret = map.size;
}

void get(HashMap map, int key, ref boolean err, ref int ret) {
    int hashMod;
    hashCode(key, hashMod);
    
    hashMod = hashMod % (map.size + MARGIN);
    if (hashMod < 0) {
        hashMod = hashMod + (map.size + MARGIN);
    }

    while (map.elementData[hashMod] != null) {
        Node node = map.elementData[hashMod];

        if (key == node.key) {
            ret = node.value;
            err = false;
            return;
        }

        hashMod = (hashMod + 1) % (map.size + MARGIN);
    }

    ret = -1;
    err = true;
}

void resize(HashMap map, int newSize, ref HashMap ret) {
    int i, h, hashMod;
    Node n;
    int k, v;

    Node[newSize + MARGIN] elementData;

    for (i = 0; i < map.size + MARGIN; i++) {
        if (map.elementData[i] != null) {
            n = map.elementData[i];
            hashMod = n.hash % (newSize + MARGIN);
            if (hashMod < 0) {
                hashMod = hashMod + (newSize + MARGIN);
            }
            while (elementData[hashMod] != null) {
                hashMod = (hashMod + 1) % (newSize + MARGIN);
            }
            elementData[hashMod] = new Node(key=n.key, value=n.value, hash=n.hash);
        }
    }

    ret = new HashMap(size = newSize, elementData = elementData);
}

void put(HashMap map, int key, int value, ref HashMap ret) {
    int hash, hashMod;
    boolean found;
    hashCode(key, hash);

    hashMod = hash % (map.size + MARGIN);
    if (hashMod < 0) {
        hashMod = hashMod + (map.size + MARGIN);
    }

    found = false;
    while (map.elementData[hashMod] != null && !found) {
        Node node = map.elementData[hashMod];
        if (key == node.key) {
            found = true;
        } else {
            hashMod = (hashMod + 1) % (map.size + MARGIN);
        }
    }

    if (map.elementData[hashMod] == null) {
        resize(map, map.size + 1, ret);

        hashMod = hash % (map.size + 1 + MARGIN);
        if (hashMod < 0) {
            hashMod = hashMod + (map.size + MARGIN + 1);
        }

        while (ret.elementData[hashMod] != null) {
            hashMod = (hashMod + 1) % (map.size + MARGIN + 1);
        }

        ret.elementData[hashMod] = new Node(key=key, value=value, hash=hash);
    } else {
        Node[map.size + MARGIN] elementData;
        elementData[0::map.size + MARGIN] = map.elementData[0::map.size + MARGIN];
        elementData[hashMod] = new Node(key=key, value=value, hash=hash);

        ret = new HashMap(size=map.size, elementData=elementData);
    }
}

<<<<<<< HEAD:more_examples/jlibsketch/HashMap/map.sp
void remove(HashMap map, int key, ref HashMap ret) {
    int hash, hashMod;
    boolean found;
    hashCode(key, hash);

    hashMod = hash % (map.size + MARGIN);
    if (hashMod < 0) {
        hashMod = hashMod + (map.size + MARGIN);
    }

    found = false;
    while (map.elementData[hashMod] != null && !found) {
        Node node = map.elementData[hashMod];
        if (key == node.key) {
            found = true;
        } else {
            hashMod = (hashMod + 1) % (map.size + MARGIN);
        }
    }

    if (!found) {
        ret = map;
        return;
    }

    int i, h, hm;
    Node n;
    int k, v;

    int newSize = map.size - 1;

    Node[newSize + MARGIN] elementData;

    for (i = 0; i < map.size + MARGIN; i++) {
        if (map.elementData[i] != null && map.elementData[i].key != key) {
            n = map.elementData[i];
            hm = n.hash % (newSize + MARGIN);
            if (hm < 0) {
                hm = hm + (newSize + MARGIN);
            }
            while (elementData[hm] != null) {
                hm = (hm + 1) % (newSize + MARGIN);
            }
            elementData[hm] = new Node(key=n.key, value=n.value, hash=n.hash);
        }
    }

    ret = new HashMap(size = newSize, elementData = elementData);
}

=======
>>>>>>> oopsla23:benchmarks/application2/HashMap/map.sp
void containsKey(HashMap map, int key, ref boolean ret) {
    int value;
    boolean err;
    
    get(map, key, err, value);
    
    ret = !err;
}