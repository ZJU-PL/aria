int main(){
  // variable declarations
  int p,q,e,a,b,d,y;
  //precondition
  assume(a==0);
  assume(b==q/2);
  assume(d==1);
  assume(y==0);
  assume(p<=q);
  assume(p>=0);
  assume(e>=0);
  // loop body
  while(e<=d){
    if (p<= a + b)
    {
        b = b/2;
        d = d/2;
    }
    else
    {
        a = a + b;
        b = b/2;
        d = d/2;
        y = y + d/2;
    }
  }
  // post-condition
  assert(  y*q-p<=0 );
}