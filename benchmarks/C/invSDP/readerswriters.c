int main(){
  // variable declarations
  int c1,c2,k0,r,w,k;
  //precondition
  assume(r==0);
  assume(w==0);
  assume(k==k0);
  assume(c1>=0);
  assume(c2>=0);
  assume(k0>=0);
  // loop body
  while(unknown()){
        if (w==0){
            r = r + 1;
            k = k - c1;
        } 
        else if (r==0)
        {
            w = w + 1;
            k = k - c2;
        }        
        else if (w==0)
        {
            r = r - 1;
            k = k + c1;
        }
  }
  // post-condition
  assert( r*c1+w*c2+k==k0);
}