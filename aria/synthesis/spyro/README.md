# Spyro[Sketch]

Spyro synthesizes provably sound, most-precise specifications from given function definitions.


## Setup

### Requirements

The artifact requires dependencies if you try to run on your local machine

* python (version >= 3.6), including
    * numpy (only for `run_benchmarks.py`)
* [Sketch](https://people.csail.mit.edu/asolar/) (version >= 1.7.6)
    * Note: The tar-ball in the Sketch wiki is 1.7.5, which may not work

### Setting Sketch Path

You should add `sketch-frontend` directory to the environment variable `$PATH`.

Alternatively, you may set path to the Sketch binary, by editing `config` file:

```
SKETCH_PATH=<PATH TO SKETCH-FRONTENT>/sketch
```



## Running Spyro[Sketch]

To run spyro on default setting, run `python3 spyro.py <PATH-TO-INPUT-FILE>`.
This will synthesize minimized properties from input file, and print the result to `stdout`.


### Flags

* `infiles`: Input files. Use concatenation of all files as the input code.
* `outfile`: Output file. Default is `stdout`
* `-v, --verbose`: Print descriptive messages, and leave all the temporary files.
* `--write-log`: Write trace log if enabled. 
* `--timeout`: Timeout of each query to Sketch. Default is 300s.
* `--disable-min`: Disable formula minimization.
* `--keep-neg-may`: Disable freezing negative examples.
* `--num-atom-max`: Number of disjuncts. Default is 3.
* `--inline-bnd`: Number of inlining/unrolling. Default is 5.

### Understanding Spyro[Sketch] input

Spyro[Sketch] takes one or more files as input and treats those files as concatenated into a single input file. Each set of input file must contain exactly one definition for `var`, `relation`, `generator`, and `example`. The following is an example of each section for list reverse function.

#### Variables
```
// Input and output variables
var {
    list l;
    list lout;
}
```

#### Signature
```
// Target functions that we aim to synthesize specifications
// The last argument to the function is output
relation {
    reverse(l, lout);
}
```

#### Property generator (i.e. search space)
```
// The DSL for the specifictions.
// It uses the top predicate as a grammar for each disjunct.
// The maximum number of disjuncts are provided by the option "--num-atom-max"
// compare is macro for { == | != | <= | >= | < | > }
// ??(n) denotes arbitrary positive integer of n bits
// Provide only input arguments to function call
generator {
    boolean AP -> is_empty(L) | !is_empty(L) 
                | equal_list(L, L) | !equal_list(L, L)
                | compare(S, S + ??(1));            
    int S -> len(L) | 0 ;
    list L -> l | lout ;
}
```

#### Example generator (i.e. example domain)
```
// recursive constructor for each type
// Provide only input arguments to function call
// integer is chosen from arbitrary positive or negative 3-bits integer
example {
    int -> ??(3) | -1 * ??(3) ;
    list -> nil() | cons(int, list);
}
```

### Implementation
```
// The return value of function is passed by reference
void reverse(list l, ref list ret) {
    if (l == null) {
        ret = null;
    } else {
        list tl_reverse;
        reverse(l.tl, tl_reverse);
        snoc(tl_reverse, l.hd, ret);
    }
}
```

### Understanding Spyro[Sketch] output

The synthesize properties are given as a code that returns a Boolean value, where the value is stored in the variable `out`. 
It means that the value stored in `out` must always be true.

For example, the following is synthesized properties of `application1/list/reverse.sp`:

```
Property 0

bit var_50 = 0;
equal_list(lout, l, var_50);
int var_52 = 0;
len(l, var_52);
bit out_s1 = var_50 || (var_52 > 1);
out = out_s1;


Property 1

int var_102 = 0;
len(lout, var_102);
int var_102_0 = 0;
len(l, var_102_0);
out = var_102 == var_102_0;
```

The property 0 means
$$eq(l_{out}, l) \vee len(l) > 1$$
must be true.

The property 1 means
$$len(l_{out}) == len(l)$$
must be true.
