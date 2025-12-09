struct E { }

struct ArrayList {
    int capacity;
    int size;
    E[capacity] elementData;
}

void newArrayList(ref ArrayList ret) {
    ret = new ArrayList(capacity = 10, size = 0);
}

void checkAdjustSize(ArrayList arr, ref ArrayList ret) {
    if (arr.size + 1 >= arr.capacity) {
        copyNewElementData(arr, arr.capacity + 10, ret);
    } else {
        copyNewElementData(arr, arr.capacity, ret);
    }
}

void copyNewElementData(ArrayList arr, int capacity, ref ArrayList ret) {
    ret = new ArrayList(capacity = capacity, size = arr.size);
    ret.elementData[0:arr.size] = arr.elementData[0:arr.size];
}

void add(ArrayList arr, E e, ref ArrayList ret) {
    checkAdjustSize(arr, ret);
    ret.elementData[ret.size++] = e;
}

void size(ArrayList arr, ref int ret) {
    ret = arr.size;
}

void get(ArrayList arr, int idx, ref E ret) {
    if (idx < 0 || idx >= arr.size) {
        ret = null;
    } else {
        ret = arr.elementData[idx];
    }
}

void removeElement(ArrayList arr, int idx, ref ArrayList ret) {
    copyNewElementData(arr, arr.capacity, ret);

    for (int j = idx; j < ret.size - 1; j++) {
        ret.elementData[j] = ret.elementData[j+1];
    }
    ret.elementData[ret.size - 1] = null;
    ret.size = ret.size - 1;
}

void remove(ArrayList arr, int idx, ref ArrayList ret) {
    if (idx < 0 || idx >= arr.size) {
        ret = arr;
    } else {
        removeElement(arr, idx, ret);
    }
}

void set(ArrayList arr, int idx, E e, ref ArrayList ret) {
    if (idx < 0 || idx >= arr.size) {
        ret = arr;
    } else {
        copyNewElementData(arr, arr.capacity, ret);
        ret.elementData[idx] = e;
    }
}

void ensureCapacity(ArrayList arr, int minCapacity, ref ArrayList ret) {
    if (minCapacity > 10) {
        ensureExplicitCapacity(arr, minCapacity, ret);
    } else {
        ret = arr;
    }
}

void ensureExplicitCapacity(ArrayList arr, int minCapacity, ref ArrayList ret) {
    if (minCapacity - arr.capacity > 0) {
        grow(arr, minCapacity, ret);
    } else {
        ret = arr;
    }
}

void grow(ArrayList arr, int minCapacity, ref ArrayList ret) {
    int oldCapacity = arr.capacity;
    int newCapacity = oldCapacity + (oldCapacity / 2);
    if (newCapacity - minCapacity < 0) { newCapacity = minCapacity; }
    if (newCapacity - 1000000 > 0) { newCapacity = 0x7fffffff; }

    copyNewElementData(arr, newCapacity, ret);
}