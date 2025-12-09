adt ArrayList {
    Empty { }
	Add { ArrayList l; KVPair e; }
    Set { ArrayList l; int idx; KVPair e; }
    Remove { ArrayList l; int idx; }
}

void rewrite_Size(ArrayList l, ref int ret) {
    switch(l) {
        case Empty: { ret = 0; }
        case Add: { 
            rewrite_Size(l.l, ret); 
            ret = ret + 1;
        }
        case Set: { rewrite_Size(l.l, ret); }
        case Remove: { 
            rewrite_Size(l.l, ret);
            if (l.idx < ret && l.idx >= 0) { ret = ret - 1; }
        }
    }
}

void rewrite_Get(ArrayList l, int idx, ref KVPair ret) {
    switch(l) {
        case Empty: { ret = null; }
        case Add: {
            int j;
            rewrite_Size(l.l, j);
            if (idx == j) {
                ret = l.e;
            } else {
                rewrite_Get(l.l, idx, ret);
            }
        }
        case Set: {
            int j;
            rewrite_Size(l.l, j);
            if (l.idx == idx && idx >= 0 && idx < j) {
                ret = l.e;
            } else {
                rewrite_Get(l.l, idx, ret);
            }
        }
        case Remove: {
            if (idx < l.idx) {
                rewrite_Get(l.l, idx, ret);
            } else {
                rewrite_Get(l.l, idx + 1, ret);
            }
        }
    }
}

void rewrite_Copy(ArrayList l, ref ArrayList ret) {
    switch(l) {
        case Empty: { ret = new Empty(); }
        case Add: {
            ArrayList ret_l;

            rewrite_Copy(l.l, ret_l);
            ret = new Add(l = ret_l, e = l.e);
        }
        case Set: {
            ArrayList ret_l;

            rewrite_Copy(l.l, ret_l);
            ret = new Set(l = ret_l, idx = l.idx, e = l.e);
        }
        case Remove: {
            ArrayList ret_l;

            rewrite_Copy(l.l, ret_l);
            ret = new Remove(l = ret_l, idx = l.idx);
        }
    }
}

void rewrite_ensureCapacity(ArrayList l, int capacity, ref ArrayList ret) {
    ret = l;
}

void list_empty(ref ArrayList ret) {
    ret = new Empty();
}

void list_add(ArrayList l, KVPair e, ref ArrayList ret) {
    ret = new Add(l = l, e = e);
}

void list_get(ArrayList l, int idx, ref KVPair ret) {
    rewrite_Get(l, idx, ret);
}

void list_set(ArrayList l, int idx, KVPair e, ref ArrayList ret) {
    ret = new Set(l = l, idx = idx, e = e);
}

void list_size(ArrayList l, ref int ret) {
    rewrite_Size(l, ret);
}

void list_remove(ArrayList l, int idx, ref ArrayList ret) {
    ret = new Remove(l = l, idx = idx);
}

void list_copy(ArrayList l, ref ArrayList ret) {
    rewrite_Copy(l, ret);
}

void list_ensureCapacity(ArrayList l, int capacity, ref ArrayList ret) {
    rewrite_ensureCapacity(l, capacity, ret);
}