struct tree_node {
    int val;
	tree_node left;
    tree_node right;	
}

void branch(int val, tree_node left, tree_node right, ref tree_node ret) {
    ret = new tree_node();
    ret.val = val;
    ret.left = left;
    ret.right = right;
}

void empty(ref tree_node ret) {
    ret = null;
}

void left(tree_node t, ref tree_node ret) {
    assert t != null;

    ret = t.left;
}

void right(tree_node t, ref tree_node ret) {
    assert t != null;

    ret = t.right;
}

void root_val(tree_node t, ref int ret) {
    assert t != null;

    ret = t.val;
}

void forall(fun f, tree_node t, ref boolean ret) {
    if (t == null) {
        ret = true;
    } else {
        boolean left_ret;
        boolean right_ret;

        forall(f, t.left, left_ret); 
        forall(f, t.right, right_ret);

        ret = left_ret && right_ret && f(t.val);
    }
}

void exists(fun f, tree_node t, ref boolean ret) {
    if (t == null) {
        ret = false;
    } else {
        boolean left_ret;
        boolean right_ret;

        exists(f, t.left, left_ret); 
        exists(f, t.right, right_ret);

        ret = left_ret || right_ret || f(t.val);
    }
}

void is_empty(tree_node t, ref boolean ret) {
    ret = (t == null);
}

void tree_equal(tree_node t1, tree_node t2, ref boolean ret) {
    if (t1 == null || t2 == null) {
        ret = t1 == t2;
    } else {
        boolean left_equal;
        boolean right_equal;
        tree_equal(t1.left, t2.left, left_equal);
        tree_equal(t1.right, t2.right, right_equal);
        ret = t1.val == t2.val && left_equal && right_equal;
    }
}

void tree_size(tree_node t, ref int ret) {
    if (t == null) {
        ret = 0;
    } else {
        int size_left;
        int size_right;
        tree_size(t.left, size_left);
        tree_size(t.right, size_right);
        ret = size_left + size_right + 1;
    }
}