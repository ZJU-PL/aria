struct queue {
	list in_list;
    list out_list;
}

void empty_queue(ref queue ret) {
    ret = new queue();
    nil(ret.in_list);
    nil(ret.out_list);
}

void is_empty_queue(queue q, ref boolean ret) {
    is_empty_list(q.out_list, ret);
}

void enqueue(queue q, int val, ref queue ret) {
    assume (q != null);

    ret = new queue();
    if (q.out_list == null) {
        ret.in_list = q.in_list;
        cons(val, q.out_list, ret.out_list);
    } else {
        cons(val, q.in_list, ret.in_list);
        ret.out_list = q.out_list;
    }
}

void dequeue(queue q, ref int val_out, ref queue q_out) {
    assume (q != null);
    assume (q.out_list != null);

    q_out = new queue();
    val_out = q.out_list.hd;
    if (q.out_list.tl == null) {
        nil(q_out.in_list);
        reverse(q.in_list, q_out.out_list);
    } else {
        q_out.in_list = q.in_list;
        q_out.out_list = q.out_list.tl;
    }
}

void queue_len(queue q, ref int ret) {
    int len_in_list;
    int len_out_list;

    list_len(q.in_list, len_in_list);
    list_len(q.out_list, len_out_list);
    ret = len_in_list + len_out_list;
}

void queue2list(queue q, ref list ret) {
    assume (q != null);

    list rev_in_list;
    reverse(q.in_list, rev_in_list);
    append(q.out_list, rev_in_list, ret);
}