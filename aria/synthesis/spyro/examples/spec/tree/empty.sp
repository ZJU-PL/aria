//@Description Binary tree

var {
    tree_node empty_out;
}

relation {
    empty(empty_out);
}

generator {
    boolean AP -> compare(S, S + ??(1)) 
                | is_empty(T) | !is_empty(T)
                | tree_equal(T, T) | !tree_equal(T, T);
    int S -> tree_size(T) | 0;
    tree_node T -> empty_out;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    tree_node(4) -> empty() | branch(int, tree_node, tree_node);
}