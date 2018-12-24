import unittest
from .Tools import solveTDM


class TestTDMA(unittest.TestCase):
    def setUp(self):
        self.a = [2, 2]
        self.b = [1, 1, 1]
        self.c = [3, 3]
        self.f = [7, 13, 7]

    def test_TDMOnEmpty(self):
        x = [0, 0, 0]
        self.assertEqual(
            list(map(round, solveTDM(self.a, self.b, self.c, x, self.f))), [1, 2, 3]
        )

    def test_TDMOnBC(self):
        x = [1, 0, 3]
        self.assertEqual(
            list(map(round, solveTDM(self.a, self.b, self.c, x, self.f))), [1, 2, 3]
        )

    def test_TDMUnaffectedObject(self):
        x = [0, 0, 0]
        self.assertEqual(x, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
