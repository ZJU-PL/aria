
int main(){
    // variable declarations
    int a,b,x,y,u,v,c;
    //precondition
    assume(a == x);
    assume(b == y);
    assume(u == b);
    assume(v == 0);
    assume(c == a*b);

    // loop body
    while(unknown()){
        if (y<=x) {
           x = x - y;
           v = v + u;
        }
        else {
            y = y -x;
            u = u + v;
        }
    }
    // post-condition
    assert( c == x*u + y*v);
}

