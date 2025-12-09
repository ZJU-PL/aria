//@Description Sketch to reverse a list.

var {
    queue empty_queue_out;
}

relation {
    empty_queue(empty_queue_out);
}

generator {
    boolean AP -> is_empty_list(L) | !is_empty_list(L)
                | equal_list(L, L) | !equal_list(L, L);
    int S -> list_len(L) | 0;
    list L -> queue2list(Q);
    queue Q -> empty_queue_out;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    queue(4) -> empty_queue() | enqueue(queue, int) | dequeue(queue, int) ;
    list -> nil() | cons(int, list);
}