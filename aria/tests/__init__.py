try:
    import unittest2 as unittest
except ImportError:
    import unittest

import logging

skipIf = unittest.skipIf


class TestCase(unittest.TestCase):
    """
    Wrapper on the unittest TestCase class.
    """

    def setUp(self):
        """
        Hook method for setting up the test fixture before exercising it.
        """
        logging.basicConfig(level=logging.DEBUG)
        pass

    def tearDown(self):
        """
        Hook method for deconstructing the test fixture after testing it.
        """
        pass

    if "assertRaisesRegex" not in dir(unittest.TestCase):
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


# Export a main function
main = unittest.main

# Export SkipTest
SkipTest = unittest.SkipTest
