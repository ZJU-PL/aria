from aria.tests import TestCase, main
from aria.bool.maxsat.bs import obv_bs


class TestOBVBS(TestCase):
    """
    Test obv-bs algorithm.
    """

    def test_obv_bs1(self):
        clauses = [[4, 2], [-2, -3]]
        literals = [3, 4, 2]
        result = obv_bs(clauses, literals)
        print(result)
        assert result == [3, 4, -2]

    def test_obv_bs2(self):
        clauses = [[4, 2], [-2, -3]]
        literals = [5, 3, 4, 2]
        result = obv_bs(clauses, literals)
        print(result)
        assert result == [5, 3, 4, -2]

    def test_obv_bs3(self):
        clauses = [[1, 2], [-2, -3, -4], [4, -5]]
        literals = [1, 3, 2, 5, 4]
        result = obv_bs(clauses, literals)
        print(result)
        assert result == [1, 3, 2, -5, -4]

    def test_obv_bs4(self):
        clauses = [[1, -2], [-1, -2]]
        literals = [2, 1, 3]
        result = obv_bs(clauses, literals)
        print(result)
        assert result == [-2, 1, 3]


if __name__ == "__main__":
    main()
