
int main(){
    // variable declarations
    int x,y,a,b,p,q,r,s;
    //precondition
    assume(x >= 1);
    assume(y >= 1);
    assume(a == x);
    assume(b == y);
    assume(p == 1);
    assume(q == 0);
    assume(r == 0);
    assume(s == 1);

    // loop body
    while(unknown()){
        if (a>=b) {
           a = a-b;
           p = p-q;
           r = r-s;
        }
        else {
           b = b-a;
           q = q-p;
           s = s-r;
        }
    }
    // post-condition
    assert((b == x*q + y*s)&&(a == y*r + x*p));
    //assert((r-s) * y == (q-p) * x)
}

