"""This module contains the PDA implementation"""

# !/usr/bin/python
from typing import Optional

try:
    print("Checking for pythonpda module:", end=" ")
    print("OK")
    from aria.automata.symautomata.pythonpda import PythonPDA, PDAState

    class PDA(PythonPDA):
        """This is the structure for a PDA"""

        def shortest_string(self) -> Optional[str]:
            """
            Uses BFS in order to find the shortest string
            Args:
                None
            Returns:
                str: The shortest string
            """
            from aria.automata.symautomata.pdadiff import PdaDiff

            ops = PdaDiff(None, None, self.alphabet)
            ops.mmc = self
            return ops.get_string()

        def diff(self, mmb):
            """
            Automata Diff operation
            """
            from aria.automata.symautomata.pdadiff import PdaDiff

            ops = PdaDiff(self, mmb, self.alphabet)
            mmc = ops.diff()
            return mmc

        # def  consume_input(self, str):
        #    """
        #    Not implemented
        #    """
        #    return 0xffff

except ImportError:
    print("FAIL")
