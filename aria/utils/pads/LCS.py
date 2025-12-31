"""Dynamic program for longest common subsequences

D. Eppstein, March 2002.
"""
# pylint: disable=invalid-name


def longest_common_subsequence(a, b):
    """Find longest common subsequence of iterables a and b."""
    a = list(a)
    b = list(b)

    # Fill dictionary lcs_len[i,j] with length of LCS of a[:i] and b[:j]
    lcs_len = {}
    for i in range(len(a)+1):
        for j in range(len(b) + 1):
            if i == 0 or j == 0:
                lcs_len[i,j] = 0
            elif a[i-1] == b[j-1]:
                lcs_len[i,j] = lcs_len[i-1,j-1] + 1
            else:
                lcs_len[i,j] = max(lcs_len[i-1,j], lcs_len[i,j-1])

    # Produce actual sequence by backtracking through pairs (i,j),
    # using computed lcs_len values to guide backtracking
    i = len(a)
    j = len(b)
    lcs = []
    while lcs_len[i,j]:
        while lcs_len[i,j] == lcs_len[i-1,j]:
            i -= 1
        while lcs_len[i,j] == lcs_len[i,j-1]:
            j -= 1
        i -= 1
        j -= 1
        lcs.append(a[i])

    lcs.reverse()
    return lcs
