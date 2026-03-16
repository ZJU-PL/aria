"""
A set of simple transition systems for testing verification algorithms
(NOTE: only for integer variables)

get_int_sys1():
    # Simple, 2-variable counter x'=x+2, and y'=y-2. 0<=x,y<=8.
       Loop back at 8 and 0. _p_x denotes primed version of x.
get_int_sys2():
    x, l, _p_x, _p_l = z3.Ints('x l x! l!')
    init = z3.And(x == 0, l == 0)
    trans = z3.Or(
get_int_sys3():
    # Simple loop: Same as above, but loop bound is symbolic(k) instead of 10.
    # Label: buggy
get_int_sys4():
    # Simple loop: Same as above, but loop bound is symbolic(k) instead of 10.
get_int_sys5():
    # Simple_vardep program from SV-COMP.
get_int_sys6():
    # Simple_vardep program from SV-COMP.
get_int_sys7():
    # A counter system with multiple variables and invariant properties.
    # x increases by 1, y by 2, and z by 3 in each step until x reaches 10.
    # The invariant property is that z = 3x and y = 2x always holds.
get_int_sys8():
    # A mutual exclusion-like system with two processes.
    # pc1, pc2 are program counters for two processes.
    # The safety property is that they cannot be in critical section simultaneously.
get_int_sys9():
    # A ticket lock system with two counters.
    # 'ticket' is the next available number
    # 'serving' is the currently served number
    # The property is that ticket is always >= serving
get_bool_sys1():
    # Original boolean test case with variables x and y
get_bool_sys2():
    # Binary counter that increments from 000 to 111
get_bool_sys3():
    # Mutual exclusion protocol with two processes
get_bool_sys4():
    # Fibonacci-like sequence with boolean variables
"""

import z3

from aria.efmc.sts import TransitionSystem


def get_int_sys1():
    """
    # Simple, 2-variable counter x'=x+2, and y'=y-2. 0<=x,y<=8.
      Loop back at 8 and 0. _p_x denotes primed version of x.
    """
    x, y, _p_x, _p_y = z3.Ints("x y x! y!")
    init = z3.And(x == 0, y == 8)
    trans = z3.Or(
        z3.And(x < 8, y <= 8, _p_x == x + 2, _p_y == y - 2),
        z3.And(x == 8, _p_x == 0, y == 0, _p_y == 8),
    )
    post = z3.Not(z3.And(x == 0, y == 0))  # Is valid.
    all_vars = [x, y, _p_x, _p_y]
    return all_vars, init, trans, post


def get_int_sys2():
    x, l, _p_x, _p_l = z3.Ints("x l x! l!")
    init = z3.And(x == 0, l == 0)
    trans = z3.Or(
        z3.And(
            l == 0,
            z3.Or(
                z3.And(x < 10, _p_x == x + 1, _p_l == l),
                z3.And(x >= 10, _p_l == 1, _p_x == x),
            ),
        ),
        z3.And(l == 1, _p_x == x, _p_l == l),
    )
    # TS with explicit limit for x, use for testing push forward.
    # Use this prop to test push fwd. Is invalid.
    post = z3.Or(z3.And(l == 1, x > 10), l == 0)
    all_vars = [x, l, _p_x, _p_l]
    return all_vars, init, trans, post


def get_int_sys3():
    """Simple loop: Same as above, but loop bound is symbolic(k) instead of 10.
    Label: buggy
    """
    x, l, k, _p_x, _p_l, _p_k = z3.Ints("x l k x! l! k!")
    init = z3.And(
        x == 0, l == 0, k >= 0
    )  # Doesn't work for init = z3.And(x==0,l==0) since k can be negative.
    trans = z3.And(
        _p_k == k,
        z3.Or(
            z3.And(
                l == 0,
                z3.Or(
                    z3.And(x < k, _p_x == x + 1, _p_l == l),
                    z3.And(x >= k, _p_l == 1, _p_x == x),
                ),
            ),
            z3.And(l == 1, _p_x == x, _p_l == l),
        ),
    )
    post = z3.Or(z3.And(l == 1, x > k), l == 0)  # This isn't valid.
    all_vars = [x, l, k, _p_x, _p_l, _p_k]
    return all_vars, init, trans, post


def get_int_sys4():
    """Simple loop: Same as above, but loop bound is symbolic(k) instead of 10."""
    x, l, k, _p_x, _p_l, _p_k = z3.Ints("x l k x! l! k!")
    init = z3.And(
        x == 0, l == 0, k >= 0
    )  # Doesn't work for init = z3.And(x==0,l==0) since k can be negative.
    trans = z3.And(
        _p_k == k,
        z3.Or(
            z3.And(
                l == 0,
                z3.Or(
                    z3.And(x < k, _p_x == x + 1, _p_l == l),
                    z3.And(x >= k, _p_l == 1, _p_x == x),
                ),
            ),
            z3.And(l == 1, _p_x == x, _p_l == l),
        ),
    )
    post = z3.Or(z3.And(l == 1, x == k), l == 0)  # This is valid!
    all_vars = [x, l, k, _p_x, _p_l, _p_k]
    return all_vars, init, trans, post


def get_int_sys5():
    """Simple_vardep program from SV-COMP."""
    i, j, k, l, _p_i, _p_j, _p_k, _p_l = z3.Ints("i j k l i! j! k! l!")
    init = z3.And(i == 0, j == 0, k == 0, l == 0)
    trans = z3.Or(
        z3.And(
            l == 0,
            z3.Or(
                z3.And(k < 100, _p_i == i + 1, _p_j == j + 2, _p_k == k + 3, _p_l == l),
                z3.And(k >= 100, _p_i == i, _p_j == j, _p_k == k, _p_l == 1),
            ),
        ),
        z3.And(l == 1, _p_i == i, _p_j == j, _p_k == k, _p_l == l),
    )
    post = z3.And(k == 3 * i, j == 2 * i)  # This is valid.
    all_vars = [i, j, k, l, _p_i, _p_j, _p_k, _p_l]
    return all_vars, init, trans, post


def get_int_sys6():
    """Simple_vardep program from SV-COMP."""
    i, j, k, l, _p_i, _p_j, _p_k, _p_l = z3.Ints("i j k l i! j! k! l!")
    init = z3.And(i == 0, j == 0, k == 0, l == 0)
    trans = z3.Or(
        z3.And(
            l == 0,
            z3.Or(
                z3.And(k < 100, _p_i == i + 1, _p_j == j + 2, _p_k == k + 3, _p_l == l),
                z3.And(k >= 100, _p_i == i, _p_j == j, _p_k == k, _p_l == 1),
            ),
        ),
        z3.And(l == 1, _p_i == i, _p_j == j, _p_k == k, _p_l == l),
    )
    post = z3.Or(l == 0, k > 3 * i)  # Not valid.
    all_vars = [i, j, k, l, _p_i, _p_j, _p_k, _p_l]
    return all_vars, init, trans, post


def get_int_sys7():
    """
    A counter system with multiple variables and invariant properties.
    x increases by 1, y by 2, and z by 3 in each step until x reaches 10.
    The invariant property is that z = 3x and y = 2x always holds.
    """
    x, y, z, _p_x, _p_y, _p_z = z3.Ints("x y z x! y! z!")
    init = z3.And(x == 0, y == 0, z == 0)
    trans = z3.Or(
        z3.And(x < 10, _p_x == x + 1, _p_y == y + 2, _p_z == z + 3),
        z3.And(x >= 10, _p_x == x, _p_y == y, _p_z == z),
    )
    post = z3.And(z == 3 * x, y == 2 * x)  # Valid invariant
    all_vars = [x, y, z, _p_x, _p_y, _p_z]
    return all_vars, init, trans, post


def get_int_sys8():
    """
    A mutual exclusion-like system with two processes.
    pc1, pc2 are program counters for two processes.
    The safety property is that they cannot be in critical section simultaneously.
    """
    pc1, pc2, _p_pc1, _p_pc2 = z3.Ints("pc1 pc2 pc1! pc2!")
    # States: 0=idle, 1=trying, 2=critical
    init = z3.And(pc1 == 0, pc2 == 0)

    # Process 1 transitions
    p1_to_try = z3.And(pc1 == 0, _p_pc1 == 1, _p_pc2 == pc2)
    p1_to_crit = z3.And(pc1 == 1, pc2 != 2, _p_pc1 == 2, _p_pc2 == pc2)
    p1_to_idle = z3.And(pc1 == 2, _p_pc1 == 0, _p_pc2 == pc2)

    # Process 2 transitions
    p2_to_try = z3.And(pc2 == 0, _p_pc2 == 1, _p_pc1 == pc1)
    p2_to_crit = z3.And(pc2 == 1, pc1 != 2, _p_pc2 == 2, _p_pc1 == pc1)
    p2_to_idle = z3.And(pc2 == 2, _p_pc2 == 0, _p_pc1 == pc1)

    trans = z3.Or(p1_to_try, p1_to_crit, p1_to_idle, p2_to_try, p2_to_crit, p2_to_idle)

    post = z3.Not(z3.And(pc1 == 2, pc2 == 2))  # Mutual exclusion property
    all_vars = [pc1, pc2, _p_pc1, _p_pc2]
    return all_vars, init, trans, post


def get_int_sys9():
    """
    A ticket lock system with two counters.
    'ticket' is the next available number
    'serving' is the currently served number
    The property is that ticket is always >= serving
    """
    ticket, serving, _p_ticket, _p_serving = z3.Ints("ticket serving ticket! serving!")
    init = z3.And(ticket == 0, serving == 0)

    # Either issue a new ticket or advance serving number
    trans = z3.Or(
        # Issue new ticket
        z3.And(_p_ticket == ticket + 1, _p_serving == serving),
        # Advance serving
        z3.And(serving < ticket, _p_serving == serving + 1, _p_ticket == ticket),
    )

    post = ticket >= serving  # Valid invariant?
    all_vars = [ticket, serving, _p_ticket, _p_serving]
    return all_vars, init, trans, post


def get_int_sys10():
    """
    A resource allocation system with bounded resources.
    'free' represents available resources
    'alloc1' and 'alloc2' represent resources allocated to two processes
    The property is that total allocated resources never exceed initial amount
    """
    free, alloc1, alloc2, _p_free, _p_alloc1, _p_alloc2 = z3.Ints(
        "free alloc1 alloc2 free! alloc1! alloc2!"
    )
    total = 10  # Total resources in system

    init = z3.And(free == total, alloc1 == 0, alloc2 == 0)

    # Allocation and deallocation transitions
    p1_alloc = z3.And(
        free >= 1, _p_free == free - 1, _p_alloc1 == alloc1 + 1, _p_alloc2 == alloc2
    )

    p2_alloc = z3.And(
        free >= 1, _p_free == free - 1, _p_alloc2 == alloc2 + 1, _p_alloc1 == alloc1
    )

    p1_dealloc = z3.And(
        alloc1 >= 1, _p_free == free + 1, _p_alloc1 == alloc1 - 1, _p_alloc2 == alloc2
    )

    p2_dealloc = z3.And(
        alloc2 >= 1, _p_free == free + 1, _p_alloc2 == alloc2 - 1, _p_alloc1 == alloc1
    )

    trans = z3.Or(p1_alloc, p2_alloc, p1_dealloc, p2_dealloc)

    post = z3.And(
        free + alloc1 + alloc2 == total,  # Resource conservation
        free >= 0,
        alloc1 >= 0,
        alloc2 >= 0,
    )  # No negative resources

    all_vars = [free, alloc1, alloc2, _p_free, _p_alloc1, _p_alloc2]
    return all_vars, init, trans, post


def get_int_sys11():
    """
    A complex system combining multiple properties to test abduction prover.
    This system models a producer-consumer scenario with bounded buffer.
    - 'produced' tracks total items produced
    - 'consumed' tracks total items consumed
    - 'buffer' tracks items in buffer (produced - consumed)
    - 'state' tracks system state (0=normal, 1=overflow, 2=underflow)

    Properties to verify:
    1. Buffer never exceeds capacity (10)
    2. Buffer is never negative
    3. If in normal state, buffer is within bounds
    4. produced >= consumed always holds
    """
    produced, consumed, buffer, state = z3.Ints("produced consumed buffer state")
    _p_produced, _p_consumed, _p_buffer, _p_state = z3.Ints(
        "produced! consumed! buffer! state!"
    )
    capacity = 10

    # Initial state
    init = z3.And(produced == 0, consumed == 0, buffer == 0, state == 0)

    # Transitions
    # Produce item (normal state)
    produce_normal = z3.And(
        state == 0,
        buffer < capacity,
        _p_produced == produced + 1,
        _p_consumed == consumed,
        _p_buffer == buffer + 1,
        _p_state == 0,
    )

    # Consume item (normal state)
    consume_normal = z3.And(
        state == 0,
        buffer > 0,
        _p_produced == produced,
        _p_consumed == consumed + 1,
        _p_buffer == buffer - 1,
        _p_state == 0,
    )

    # Attempt to produce when buffer full (causes overflow)
    produce_overflow = z3.And(
        state == 0,
        buffer == capacity,
        _p_produced == produced + 1,
        _p_consumed == consumed,
        _p_buffer == buffer + 1,
        _p_state == 1,  # Overflow state
    )

    # Attempt to consume when buffer empty (causes underflow)
    consume_underflow = z3.And(
        state == 0,
        buffer == 0,
        _p_produced == produced,
        _p_consumed == consumed + 1,
        _p_buffer == buffer - 1,
        _p_state == 2,  # Underflow state
    )

    # Recovery from overflow state
    recover_overflow = z3.And(
        state == 1,
        _p_produced == produced,
        _p_consumed == consumed + 1,
        _p_buffer == buffer - 1,
        _p_state == z3.If(buffer - 1 <= capacity, 0, 1),
    )

    # Recovery from underflow state
    recover_underflow = z3.And(
        state == 2,
        _p_produced == produced + 1,
        _p_consumed == consumed,
        _p_buffer == buffer + 1,
        _p_state == z3.If(buffer + 1 >= 0, 0, 2),
    )

    # Combined transition relation
    trans = z3.Or(
        produce_normal,
        consume_normal,
        produce_overflow,
        consume_underflow,
        recover_overflow,
        recover_underflow,
    )

    # Safety properties
    # We want to verify that:
    # 1. If in normal state, buffer is within bounds
    # 2. produced >= consumed always holds
    post = z3.And(
        z3.Implies(state == 0, z3.And(buffer >= 0, buffer <= capacity)),
        produced >= consumed,
    )

    all_vars = [
        produced,
        consumed,
        buffer,
        state,
        _p_produced,
        _p_consumed,
        _p_buffer,
        _p_state,
    ]
    return all_vars, init, trans, post


def get_int_sys12():
    """
    A system with an invalid property to test how the AbductionProver handles unsafe systems.
    This is a modified version of get_int_sys7 where the property is intentionally invalid.

    The system has three counters (x, y, z) where:
    - x increases by 1 in each step
    - y increases by 2 in each step
    - z increases by 3 in each step

    The invalid property claims that z = 2*x (which is false, since z = 3*x is the correct relation)
    """
    x, y, z, _p_x, _p_y, _p_z = z3.Ints("x y z x! y! z!")
    init = z3.And(x == 0, y == 0, z == 0)
    trans = z3.Or(
        z3.And(x < 10, _p_x == x + 1, _p_y == y + 2, _p_z == z + 3),
        z3.And(x >= 10, _p_x == x, _p_y == y, _p_z == z),
    )
    # Invalid property (z should be 3*x, not 2*x)
    post = z3.And(z == 2 * x, y == 2 * x)
    all_vars = [x, y, z, _p_x, _p_y, _p_z]
    return all_vars, init, trans, post


def create_simple_sts(use_real=True):
    """
    Create a simple transition system for testing purposes.

    Args:
        use_real: If True, use real variables, otherwise use integer variables

    Returns:
        A simple TransitionSystem instance
    """
    sts = TransitionSystem()

    # Create variables based on the type
    if use_real:
        x, y = z3.Reals("x y")
        xp, yp = z3.Reals("x! y!")
    else:
        x, y = z3.Ints("x y")
        xp, yp = z3.Ints("x! y!")

    # Set up the transition system
    sts.variables = [x, y]
    sts.prime_variables = [xp, yp]

    # Initial condition: x >= 0 and y >= 0
    sts.init = z3.And(x >= 0, y >= 0)

    # Transition relation: x' = x + 1, y' = y + x
    sts.trans = z3.And(xp == x + 1, yp == y + x)

    # Post-condition: y >= 0
    sts.post = y >= 0

    return sts


# Boolean transition systems from bp_pred_abs.py


def get_bool_sys1():
    """
    Original boolean test case with variables x and y.
    """
    x, y, xp, yp = z3.Bools("x y x! y!")
    init = z3.And(x, y)
    trans = z3.And(
        z3.Implies(y, z3.And(xp == y, yp == y)),
        z3.Implies(z3.Not(y), z3.And(xp == z3.Not(y), yp == y)),
    )
    post = x
    all_vars = [x, y, xp, yp]
    return all_vars, init, trans, post


def get_bool_sys2():
    """
    Binary counter that increments from 000 to 111 in binary.
    """
    a, b, c, ap, bp, cp = z3.Bools("a b c a! b! c!")

    # Counter that increments from 000 to 111 in binary
    init = z3.And(z3.Not(a), z3.Not(b), z3.Not(c))
    trans = z3.And(ap == z3.Xor(a, z3.And(b, c)), bp == z3.Xor(b, c), cp == z3.Not(c))
    post = z3.Implies(z3.And(a, b, c), z3.And(a, b, c))  # Trivially true

    all_vars = [a, b, c, ap, bp, cp]
    return all_vars, init, trans, post


def get_bool_sys3():
    """
    Mutual exclusion protocol with two processes.
    """
    p, q, p_crit, q_crit, pp, qp, p_critp, q_critp = z3.Bools(
        "p q p_crit q_crit p! q! p_crit! q_crit!"
    )

    # Simple mutual exclusion protocol
    init = z3.And(z3.Not(p_crit), z3.Not(q_crit), z3.Not(p), z3.Not(q))

    # Transition relation for a simple mutex protocol
    trans = z3.And(
        # Process p
        z3.Implies(
            z3.And(z3.Not(p), z3.Not(p_crit)),
            z3.And(pp == True, p_critp == z3.Not(q_crit)),
        ),
        z3.Implies(z3.And(p, z3.Not(p_crit)), z3.And(pp == False, p_critp == True)),
        z3.Implies(p_crit, z3.And(pp == False, p_critp == False)),
        # Process q
        z3.Implies(
            z3.And(z3.Not(q), z3.Not(q_crit)),
            z3.And(qp == True, q_critp == z3.Not(p_crit)),
        ),
        z3.Implies(z3.And(q, z3.Not(q_crit)), z3.And(qp == False, q_critp == True)),
        z3.Implies(q_crit, z3.And(qp == False, q_critp == False)),
    )

    # Mutual exclusion property
    post = z3.Not(z3.And(p_crit, q_crit))

    all_vars = [p, q, p_crit, q_crit, pp, qp, p_critp, q_critp]
    return all_vars, init, trans, post


def get_bool_sys4():
    """
    Fibonacci-like sequence with boolean variables.
    """
    i, j, ip, jp = z3.Bools("i j i! j!")

    # Initialize i and j to false
    init = z3.And(z3.Not(i), z3.Not(j))

    # Transition: i' = j, j' = i XOR j
    trans = z3.And(ip == j, jp == z3.Xor(i, j))

    # Property: i and j are never both true simultaneously
    post = z3.Not(z3.And(i, j))

    all_vars = [i, j, ip, jp]
    return all_vars, init, trans, post
