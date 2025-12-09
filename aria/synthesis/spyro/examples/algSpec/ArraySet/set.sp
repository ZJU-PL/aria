struct ArraySet {
    int size;
    int[size] set;
}

void newArraySet(ref ArraySet ret) {
    int size = 0;
    int[size] set;

    ret = new ArraySet(
        size = size,
        set = set
    );
}

void equal(ArraySet s1, ArraySet s2, ref boolean ret) {
    assert s1 != null;
    assert s2 != null;

    if (s1.size != s2.size) {
        ret = false;
        return;
    }

    boolean b;
    for (int i=0; i < s1.size; i++) {
        contains(s2, s1.set[i], b);
        if (!b) {
            ret = false;
            return;
        }
    }
    ret = true;
}

void getIndex(ArraySet arr, int v, ref int ret) {
    int i = 0;
    for (i=0; i<arr.size; i++) {
        if (arr.set[i] == v) {
            ret = i;
            return;
        }
    }
    ret = -1;
}

void contains(ArraySet arr, int v, ref boolean ret) {
    int idx;
    getIndex(arr, v, idx);
    ret = idx >= 0;
}

void add(ArraySet arr, int e, ref ArraySet ret) {
    boolean contains_e;
    contains(arr, e, contains_e);

    if(contains_e) {
        ret = arr;
    } else {
        int size = arr.size + 1;
        int[size] set;

        set[0::arr.size] = arr.set[0::arr.size];
        set[arr.size] = e;
        
        ret = new ArraySet(
            size = size,
            set = set
        );
    }
}

void size(ArraySet arr, ref int ret) {
    ret = arr.size;
}

void remove(ArraySet arr, int e, ref ArraySet ret) {
    int idx;
    getIndex(arr, e, idx);
    if (idx >= 0) {
        int[arr.size-1] set;
        set[0::idx] = arr.set[0::idx];
        set[idx::(arr.size-idx-1)] = arr.set[(idx+1)::(arr.size-idx-1)];
        
        ret = new ArraySet(
            size = arr.size-1,
            set = set
        );
    } else {
        ret = arr;
    }
}