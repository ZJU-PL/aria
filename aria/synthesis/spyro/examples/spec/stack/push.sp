//@Description Sketch to reverse a list.

var {
    stack s;
    int val;
    stack s_out;
}

relation {
    push(s, val, s_out);
}

generator {
    boolean AP -> is_empty(ST) | !is_empty(ST)
                | stack_equal(ST, ST) | !stack_equal(ST, ST)
                | compare(S, S + ??(1));
    int S -> stack_len(ST) | 0;
    stack ST -> s | s_out;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    stack -> empty() | push(stack, int);
}