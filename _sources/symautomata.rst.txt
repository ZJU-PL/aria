Symbolic Finite Automata
========================

The ``aria.automata.symautomata`` module provides a comprehensive implementation of symbolic finite automata and related computational models. This module supports various types of automata including Deterministic Finite Automata (DFA), Symbolic Finite Automata (SFA), and Pushdown Automata (PDA).

Overview
--------

Symbolic finite automata extend traditional finite automata by allowing transitions to be labeled with predicates over potentially infinite alphabets, rather than just individual symbols. This makes them particularly useful for string analysis, regular expression processing, and formal verification tasks.

Core Components
---------------

DFA (Deterministic Finite Automata)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DFA implementation provides multiple backends for maximum compatibility:

.. automodule:: aria.automata.symautomata.dfa
   :members:
   :undoc-members:
   :show-inheritance:

**Key Features:**

* **Multiple Backend Support**: Automatically selects the best available backend:
  
  - ``pywrapfst``: OpenFST Python bindings (preferred)
  - ``fst``: Alternative FST library
  - Pure Python implementation (fallback)

* **Core Operations**:
  
  - Minimization and determinization
  - Boolean operations (union, intersection, complement, difference)
  - Regular expression conversion
  - String acceptance testing
  - Shortest string generation

**Example Usage:**

.. code-block:: python

   from aria.automata.symautomata.dfa import DFA
   from aria.automata.symautomata.alphabet import createalphabet
   
   # Create a DFA with default alphabet
   dfa = DFA(createalphabet())
   
   # Add states and transitions
   dfa.add_arc(0, 1, 'a')
   dfa.add_arc(1, 2, 'b')
   dfa[2].final = True
   
   # Test string acceptance
   result = dfa.consume_input("ab")  # Returns True
   
   # Convert to regular expression
   regex = dfa.to_regex()
   
   # Find shortest accepted string
   shortest = dfa.shortest_string()

SFA (Symbolic Finite Automata)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.sfa
   :members:
   :undoc-members:
   :show-inheritance:

Symbolic Finite Automata extend traditional DFAs by using predicates on transitions instead of individual symbols. This allows for more compact representations when dealing with large or infinite alphabets.

**Key Components:**

* **Predicates**: Abstract conditions that determine transition validity
* **SetPredicate**: Concrete implementation using character sets
* **SFA States and Arcs**: Extended state and transition structures

**Example Usage:**

.. code-block:: python

   from aria.automata.symautomata.sfa import SFA, SetPredicate
   import string
   
   # Create SFA for pattern matching
   sfa = SFA(list(string.ascii_lowercase))
   
   # Add transitions with predicates
   # Accept any lowercase letter except 'x'
   not_x = SetPredicate([c for c in string.ascii_lowercase if c != 'x'])
   sfa.add_arc(0, 0, not_x)
   
   # Accept 'x' to go to final state
   x_only = SetPredicate(['x'])
   sfa.add_arc(0, 1, x_only)
   sfa.states[1].final = True
   
   # Convert to concrete DFA
   dfa = sfa.concretize()

PDA (Pushdown Automata)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.pda
   :members:
   :undoc-members:
   :show-inheritance:

Pushdown Automata extend finite automata with a stack, enabling recognition of context-free languages.

**Features:**

* Context-free language recognition
* Stack-based computation model
* Integration with CFG (Context-Free Grammar) processing

Utility Modules
---------------

Alphabet Management
~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.alphabet
   :members:
   :undoc-members:
   :show-inheritance:

The alphabet module provides flexible alphabet creation and management:

.. code-block:: python

   from aria.automata.symautomata.alphabet import createalphabet
   
   # Default printable ASCII alphabet
   alpha1 = createalphabet()
   
   # Custom range (Unicode code points)
   alpha2 = createalphabet("65-91,97-123")  # A-Z, a-z
   
   # From file
   alpha3 = createalphabet("my_alphabet.txt")

Regular Expression Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.regex
   :members:
   :undoc-members:
   :show-inheritance:

The regex module implements the Brzozowski algebraic method for converting DFAs to regular expressions:

.. code-block:: python

   from aria.automata.symautomata.regex import Regex
   
   # Convert DFA to regex
   converter = Regex(my_dfa)
   regex_string = converter.get_regex()

Brzozowski Algorithm
~~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.brzozowski
   :members:
   :undoc-members:
   :show-inheritance:

Implements Brzozowski's algebraic method for regular expression derivation:

.. code-block:: python

   from aria.automata.symautomata.brzozowski import Brzozowski
   
   # Apply Brzozowski algorithm
   brz = Brzozowski(input_dfa)
   regex_map = brz.get_regex()

Advanced Features
----------------

Context-Free Grammar Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.cfggenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.symautomata.cfgpda
   :members:
   :undoc-members:
   :show-inheritance:

The module includes comprehensive support for context-free grammars:

* **CFG to CNF conversion**: Transform context-free grammars to Chomsky Normal Form
* **CFG to PDA conversion**: Convert grammars to equivalent pushdown automata
* **String generation**: Generate strings from CFG rules

Flex Integration
~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.flex2fst
   :members:
   :undoc-members:
   :show-inheritance:

Integration with Flex (Fast Lexical Analyzer) for converting lexical specifications to finite automata:

.. code-block:: python

   from aria.automata.symautomata.flex2fst import Flexparser
   
   # Parse Flex file to DFA
   parser = Flexparser(["a", "b", "c"])
   dfa = parser.yyparse("lexer.l")

Implementation Backends
----------------------

The module provides multiple implementation backends for different use cases:

Pure Python Backend
~~~~~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.pythondfa
   :members:
   :undoc-members:
   :show-inheritance:

* **Advantages**: No external dependencies, full Python compatibility
* **Use cases**: Development, testing, educational purposes
* **Features**: Complete DFA operations, Hopcroft minimization

OpenFST Backend
~~~~~~~~~~~~~~

.. automodule:: aria.automata.symautomata.pywrapfstdfa
   :members:
   :undoc-members:
   :show-inheritance:

* **Advantages**: High performance, industry-standard library
* **Requirements**: OpenFST library with Python bindings
* **Features**: Optimized operations, large-scale automata support

String Analysis
--------------

.. automodule:: aria.automata.symautomata.pdastring
   :members:
   :undoc-members:
   :show-inheritance:

Advanced string analysis capabilities using state removal algorithms:

* **State removal**: Systematic elimination of states to derive regular expressions
* **Path analysis**: Trace execution paths through automata
* **String generation**: Generate strings accepted by automata

Performance Considerations
-------------------------

Backend Selection
~~~~~~~~~~~~~~~~

The module automatically selects the best available backend:

1. **pywrapfst**: Preferred for performance-critical applications
2. **fst**: Alternative FST implementation
3. **pythondfa**: Fallback pure Python implementation

For optimal performance:

* Install OpenFST with Python bindings for large automata
* Use the pure Python backend for small automata or development
* Consider alphabet size when choosing predicates vs. explicit transitions

Memory Usage
~~~~~~~~~~~

* **Symbolic representations**: Use SFA for large alphabets
* **Minimization**: Apply minimization to reduce state count
* **Alphabet optimization**: Use custom alphabets to reduce memory footprint

Common Patterns
--------------

Pattern Matching
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create DFA for email validation pattern
   email_dfa = DFA()
   # ... build email validation automaton
   
   # Test email addresses
   valid = email_dfa.consume_input("user@domain.com")

Language Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Language intersection
   result_dfa = dfa1.intersect(dfa2)
   
   # Language union
   union_dfa = dfa1.union(dfa2)
   
   # Language complement
   complement_dfa = dfa1.complement(alphabet)
   
   # Language difference
   diff_dfa = dfa1.difference(dfa2)

Regular Expression Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert automaton to regex
   from aria.automata.symautomata.regex import Regex
   
   converter = Regex(my_dfa)
   regex_pattern = converter.get_regex()

Error Handling
-------------

The module includes robust error handling:

* **Import errors**: Graceful fallback between backends
* **Invalid operations**: Clear error messages for malformed automata
* **Resource limits**: Appropriate handling of large automata

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Backend not available**: Install required dependencies (OpenFST, pywrapfst)
2. **Memory issues**: Use minimization and appropriate alphabet sizes
3. **Performance problems**: Consider backend selection and automaton size

Dependencies
~~~~~~~~~~~

* **Required**: Python 3.6+
* **Optional**: OpenFST library, pywrapfst, networkx (for visualization)
* **Development**: pytest, sphinx (for documentation)

API Reference
------------

For detailed API documentation, see the individual module documentation above. The main entry points are:

* ``DFA``: Main deterministic finite automaton class
* ``SFA``: Symbolic finite automaton class  
* ``PDA``: Pushdown automaton class
* ``createalphabet()``: Alphabet creation utility
* ``Regex``: Regular expression conversion utility
