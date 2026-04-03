int main(){
  // variable declarations
  int a,e,r,q,p;
  //precondition
  assume(r==a-1);
  assume(q==1);
  assume(p==0.5);
  assume(e>=0);
  assume(a>=1);
  // loop body
  while(e <= 2*p*r){
    if ( p + 2*q <= 2*r)
    {
        r = 2*r - 2*q - p;
        q = q + p;
        p = p/2;
    }
    else
    {
      r = 2*r;
      p = p/2;
    }
  }
  // post-condition
  assert(  a-e<=q*q );
}