implementation Counter() {
    var x, count: int;
    
    entry:
        assume x >= 0;
        count := 0;
        goto loop_header;
    
    loop_header:
        goto loop_body, loop_exit;
    
    loop_body:
        assume x > 0;
        x := x - 1;
        count := count + 1;
        goto loop_header;
    
    loop_exit:
        assume x <= 0;
        assert count >= 0;
        return;
}
