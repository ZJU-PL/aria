//@Description Sketch to reverse a list.

var {
    stack empty_out;
}

relation {
    empty(empty_out);
}

generator {
    boolean AP -> is_empty(ST) | !is_empty(ST)
                | compare(S, S + ??(1));
    int S -> stack_len(ST) | 0;
    stack ST -> empty_out;
}


example {
    int -> ??(3) | -1 * ??(3) ;
    stack -> empty() | push(stack, int);
}