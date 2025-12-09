//@Description Sketch to reverse a list.

var {
    stack push_s;
    int push_val;
    stack push_s_out;

    stack pop_s;
    stack pop_s_out;
    int pop_val_out;
}

relation {
    push(push_s, push_val, push_s_out);
    pop(pop_s, pop_s_out, pop_val_out);
}

generator {
    boolean AP -> compare(I, I)
                | is_empty(ST) | !is_empty(ST)
                | stack_equal(ST, ST) | !stack_equal(ST, ST);
    int I -> push_val | pop_val_out;
    stack ST -> push_s | push_s_out | pop_s | pop_s_out;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    stack -> empty() | push(stack, int);
}