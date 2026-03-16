int main(){
  // variable declarations
  int m,s,e,i,o,t;
  //precondition
  assume(m==0);
  assume(s==0);
  assume(e==0);
  assume(t==i+o);
  // loop body
  while(unknown()){
        if (i>=1){
            m = 0;
            s = s + e + 1;
            e = 0;
            i = i - 1;
            o = o + m;
        } 
        else if (e>=1)
        {
            m = m + 1;
            e = e - 1;
        }        
        else if (s>=1)
        {
            m = 0;
            s = 0;
            e = 1;
            i = i + m + e + s + o - 1;
            o = 0;
        }
        else
        {
            m = 0;
            s = 0;
            e = 1;
            i = i + m + e + s + o - 1;
            o = 0;
        }
    }
  // post-condition
  assert( t == m + s + e + i + o);
}