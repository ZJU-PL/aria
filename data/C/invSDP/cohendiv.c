int main(){
  // variable declarations
  int x,y,d,r,q;
  //precondition
  assume(r==x);
  assume(q==0);
  assume(d>=0);
  assume(x>=0 && y>=0);
  // loop body
  while(y <= r){
    r = r - y * d;
    q = q + d;
  }
  // post-condition
  assert( x == q * y + r);
}