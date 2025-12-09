//@Description Binary tree

var {
    int val;
    tree_node t_left;
    tree_node t_right;
    tree_node t_out;

    tree_node right_in;
    tree_node right_out;
}

relation {
    branch(val, t_left, t_right, t_out);
    right(right_in, right_out);
}

generator {
    boolean AP -> is_empty(T) | !is_empty(T)
                | tree_equal(T, T) | !tree_equal(T, T);
    tree_node T -> t_left | t_right | t_out | right_in | right_out;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    tree_node(4) -> empty() | branch(int, tree_node, tree_node);
}