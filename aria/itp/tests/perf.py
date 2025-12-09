if __name__ == "__main__":
    import aria.itp as kd
    import aria.itp.config as config
    config.timing = True
    import time
    start_time = time.perf_counter()
    modules = []
    def mark(tag):
        global start_time
        elapsed_time = time.perf_counter() - start_time
        modules.append((elapsed_time, tag))
        start_time = time.perf_counter()
    start_time = time.perf_counter()
    import aria.itp.all
    mark("aria.itp.all")
    import aria.itp.theories.real as R
    mark("real")
    import aria.itp.theories.bitvec as bitvec
    mark("bitvec")
    import aria.itp.theories.real.complex as complex
    mark("complex")
    import aria.itp.theories.algebra.group as group
    mark("group")
    import aria.itp.theories.algebra.lattice
    mark("lattice")
    import aria.itp.theories.algebra.ordering
    mark("ordering")
    import aria.itp.theories.bool as bool_
    mark("bool")
    import aria.itp.theories.int
    mark("int")
    import aria.itp.theories.real.interval
    mark("interval")
    import aria.itp.theories.seq as seq
    mark("seq")
    import aria.itp.theories.set
    mark("set")
    import aria.itp.theories.fixed
    mark("fixed")
    import aria.itp.theories.float
    mark("float")
    import aria.itp.theories.real.arb
    mark("arb")
    import aria.itp.theories.real.sympy
    mark("sympy")
    import aria.itp.theories.nat
    mark("nat")
    import aria.itp.theories.real.vec
    mark("vec")
    import aria.itp.theories.logic.intuitionistic
    mark("intuitionistic")
    import aria.itp.theories.logic.temporal
    mark("temporal")

    print("\n========= Module import times ========\n")
    for (elapsed_time, tag) in sorted(modules, reverse=True):
        print(f"{elapsed_time:.6f} {tag}")

    import itertools
    for tag, group in itertools.groupby(sorted(config.perf_log, key=lambda x: x[0]), key=lambda x: x[0]):
        print("\n=============" + tag + "=============\n")
        for (tag, data, time) in sorted(group, key=lambda x: x[2], reverse=True)[:20]:
            print(f"{time:.6f}: {data}")
