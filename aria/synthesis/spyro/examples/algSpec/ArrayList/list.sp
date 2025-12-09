struct ArrayList {
    int size;
    int[size] elementData;
}

void newArrayList(ref ArrayList ret) {
    int size = 0;
    int[size] elementData;

    ret = new ArrayList(
        size = size,
        elementData = elementData
    );
}

void add(ArrayList arr, int e, ref ArrayList ret) {
    int size = arr.size + 1;

    int[size] elementData;

    elementData[0::arr.size] = arr.elementData[0::arr.size];
    elementData[arr.size] = e;
    
    ret = new ArrayList(
        size = size,
        elementData = elementData
    );
}

void size(ArrayList arr, ref int ret) {
    ret = arr.size;
}

void get(ArrayList arr, int idx, ref int ret) {
    if (idx < 0 || idx >= arr.size) {
        ret = -1; // -1 for error
    } else {
        ret = arr.elementData[idx];
    }
}