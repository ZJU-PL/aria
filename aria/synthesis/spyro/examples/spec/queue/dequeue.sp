//@Description Sketch to reverse a list.

var {
    queue q;
    int val_out;
    queue q_out;
}

relation {
    dequeue(q, val_out, q_out);
}

generator {
    boolean AP -> is_empty_list(L) | !is_empty_list(L)
                | equal_list(L, L) | !equal_list(L, L);
    int I -> val_out;
    list L -> AL | cons(I, AL) | snoc(AL, I);
    list AL -> queue2list(Q);
    queue Q -> q | q_out;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    queue(3) -> empty_queue() | enqueue(queue, int) | dequeue(queue, int) ;
    list(3) -> nil() | cons(int, list);
}
