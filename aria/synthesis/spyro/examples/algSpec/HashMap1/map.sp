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

struct HashNode {
    Key key;
    Value value;
    HashNode next;
}

struct HashTable {
    int size;
    int initialCapacity;
    int currentCapacity;
    ArrayList buckets;
}

void newHashTable(ref HashTable ret) {
    ret = new HashTable();

    ret.size = 0;
    ret.initialCapacity = 3;
    ret.currentCapacity = 3;
    list_empty(ret.buckets);

    for (int i = 0; i < 3; i++) {
        ArrayList buckets_tmp;
        list_add(ret.buckets, null, buckets_tmp);
        ret.buckets = buckets_tmp;
    }
}

void clear(HashTable tb, ref HashTable ret) {
    ret = new HashTable();

    ret.size = 0;
    ret.initialCapacity = tb.initialCapacity;
    ret.currentCapacity = ret.initialCapacity;
    list_empty(ret.buckets);

    for (int i = 0; i < ret.currentCapacity; i++) {
        ArrayList buckets_tmp;
        list_add(ret.buckets, null, buckets_tmp);
        ret.buckets = buckets_tmp;
    }   
}

void size(HashTable tb, ref int ret) {
    ret = tb.size;
}

void get(HashTable tb, Key key, ref Value ret) {
    HashNode result;
    getNodeWithKey(tb, key, result);

    if (result != null) {
        ret = result.value;
    } else {
        ret = null;
    }
}

void getBucketIndex(HashTable tb, Key key, ref int ret) {
    int h = key.hash;
    ret = h % tb.currentCapacity;
    if (ret < 0) { ret = -1 * ret; }
}

void getNodeWithKey(HashTable tb, Key key, ref HashNode ret) {
    if (tb.size == 0 || key == null) {
        ret = null;
        return;
    }

    int bucketIndex;
    getBucketIndex(tb, key, bucketIndex);

    HashNode curr;
    list_get(tb.buckets, bucketIndex, curr);

    while (curr != null) {
        Key k = curr.key;
        boolean b;
        equalKey(k, key, b);

        if (b) {
            ret = curr;
            return;
        }

        curr = curr.next;
    }

    ret = null;
}

void getNodeWithValue(HashTable tb, Value value, ref HashNode ret) {
    if (tb.size == 0) {
        ret = null;
        return;
    }

    int s;
    list_size(tb.buckets, s);
    for (int i = 0; i < s; i++) {
        HashNode curr;
        list_get(tb.buckets, i, curr);

        while (curr != null) {
            Value v = curr.value;
            boolean b;
            equalValue(v, value, b);

            if (b) {
                ret = curr;
                return;
            }

            curr = curr.next;
        }
    }

    ret = null;
}

void containsKey(HashTable tb, Key key, ref boolean ret) {
    HashNode result;
    getNodeWithKey(tb, key, result);

    ret = result != null;
}

void containsValue(HashTable tb, Value value, ref boolean ret) {
    HashNode result;
    getNodeWithValue(tb, value, result);

    ret = result != null;
}

void putSub(HashTable tb, Key key, Value value) {
    int bucketIndex;
    getBucketIndex(tb, key, bucketIndex);

    HashNode newNode = new HashNode(key=key, value=value, next=null);
    HashNode current;
    list_get(tb.buckets, bucketIndex, current);

    // If bucket is empty, set as first node and we're done
    if (current == null) {
        ArrayList result;
        list_set(tb.buckets, bucketIndex, newNode, result);
        tb.buckets = result;
        tb.size = tb.size + 1;
        return;
    } 
    
    // Traverse the list within the bucket until match or end found
    while (current != null) {
        // When a key match is found, replace the value it stores and break
        boolean b;
        equalKey(key, current.key, b);
        if (b) {
            current.value = value;
            return;
        }

        // When the last node of the list is reached, append new node here and break
        else if (current.next == null) {
            current.next = newNode;
            tb.size = tb.size + 1;
            return;
        }

        current = current.next;
    }  
}

void put(HashTable tb, Key key, Value value, ref HashTable ret) {
    ensureCapacity(tb, tb.size + 1, ret);

    putSub(ret, key, value);
}

void node_copy(HashNode n, ref HashNode ret) {
    if (n == null) {
        ret = null;
    } else {
        ret = new HashNode();
        ret.key = n.key;
        ret.value = n.value;
        node_copy(n.next, ret.next);
    }
}

void table_copy(HashTable tb, ref HashTable ret) {
    ret = new HashTable();
    ret.size = tb.size;
    ret.initialCapacity = tb.initialCapacity;
    ret.currentCapacity = tb.currentCapacity;
    list_copy(tb.buckets, ret.buckets);
}

void remove(HashTable tb, Key key, ref HashTable ret) {
    if (tb.size == 0 || key == null) {
        ret = tb;
    }

    table_copy(tb, ret);

    int bucketIndex;
    getBucketIndex(ret, key, bucketIndex);

    HashNode current;
    HashNode previous = null;
    list_get(ret.buckets, bucketIndex, current);   

    // Traverse the list inside the bucket until match is found or end of list reached
    while (current != null) {
        boolean b;
        equalKey(key, current.key, b);
        if (b) {
            // Handle case when node is first in bucket
            if (previous == null) {
                // If there is a next node, set next node as first in bucket
                if (current.next != null) {
                    ArrayList buckets;
                    list_set(ret.buckets, bucketIndex, current.next, buckets);
                    ret.buckets = buckets;
                } 
                // If there is no other node in list, simply set bucket to null
                else {
                    ArrayList buckets;
                    list_set(ret.buckets, bucketIndex, null, buckets);
                    ret.buckets = buckets;
                }
            } 
            // Handle case when node is not in first position
            else {
                // If it's the last node in the list, set previous' next as null
                if (current.next == null) {
                    previous.next = null;
                }
                // If it's anywhere else in the list, connect previous and next
                else {
                    previous.next = current.next;
                }
            }
    
            ret.size = ret.size - 1;
            return;
        }

        previous = current;
        current = current.next;
    }   
}

void ensureCapacity(HashTable tb, int intendedCapacity, ref HashTable ret) {
    // Within the load limit (3/4)
    if (intendedCapacity * 4 < tb.currentCapacity * 3) {
        table_copy(tb, ret);
        return;
    }

    int newCapacity = tb.currentCapacity * 2;
    ArrayList buckets;
    list_ensureCapacity(tb.buckets, newCapacity, buckets);

    for (int i = 0; i < newCapacity; i++) {
        ArrayList buckets_tmp;
        list_add(buckets, null, buckets_tmp);
        buckets = buckets_tmp;    
    }

    ret = new HashTable();

    ret.buckets = buckets;
    ret.initialCapacity = tb.initialCapacity;
    ret.currentCapacity = newCapacity;
    ret.size = 0;

    for (int i = 0; i < tb.currentCapacity; i++) {
        HashNode current;
        list_get(tb.buckets, i, current);
        while (current != null) {
            putSub(ret, current.key, current.value);
            current = current.next;
        }
    }
}

void equalMap(HashTable tb1, HashTable tb2, ref boolean ret) {
    if (tb1.size != tb2.size) {
        ret = false;
        return;
    }

    int index = 0;
    for (int i = 0; i < tb1.size; i++) {
        HashNode current;
        list_get(tb1.buckets, i, current);
        while (current != null) {
            Value val;
            boolean b;

            get(tb2, current.key, val);
            equalValue(current.value, val, b);

            if (!b) {
                ret = false;
                return;
            }

            index = index + 1;
            current = current.next;
        }
    }

    ret = true;
}