"""Find longest increasing subsequence of an input sequence.
D. Eppstein, April 2004
"""
# pylint: disable=invalid-name

import unittest
from bisect import bisect_left

def longest_increasing_subsequence(seq):
    """
    Find and return longest increasing subsequence of seq.
    If multiple increasing subsequences exist, the one that ends
    with the smallest value is preferred, and if multiple
    occurrences of that value can end the sequence, then the
    earliest occurrence is preferred.
    """

    # The main data structures: head[i] is value x from seq that
    # terminates a length-i subsequence, and tail[i] is either
    # None (if i=0) or the pair (head[i-1],tail[i-1]) as they
    # existed when x was processed.
    head = []
    tail = [None]

    for x in seq:
        i = bisect_left(head,x)
        if i >= len(head):
            head.append(x)
            if i > 0:
                tail.append((head[i-1],tail[i-1]))
        elif head[i] > x:
            head[i] = x
            if i > 0:
                tail[i] = head[i-1],tail[i-1]

    if not head:
        return []

    output = [head[-1]]
    pair = tail[-1]
    while pair is not None:
        # pair is guaranteed to be a tuple when not None
        # pylint: disable=unpacking-non-sequence
        x, pair = pair
        output.append(x)

    output.reverse()
    return output

# If run as "python LongestIncreasingSubsequence.py", run tests on various
# small lists and check that the correct subsequences are generated.

class LISTest(unittest.TestCase):
    def test_lis(self):
        self.assertEqual(longest_increasing_subsequence([]),[])
        self.assertEqual(longest_increasing_subsequence(range(10,0,-1)),[1])
        self.assertEqual(longest_increasing_subsequence(range(10)),range(10))
        self.assertEqual(
            longest_increasing_subsequence([3,1,4,1,5,9,2,6,5,3,5,8,9,7,9]),
            [1,2,3,5,8,9])

if __name__ == "__main__":
    unittest.main()
