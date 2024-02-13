from src.util import generate_four_digit_NACA
from scipy import stats
import numpy as np


def test_generate_four_digit_NACA():
    data = np.loadtxt('airfoils/NACA2412.txt')
    XB = data[:, 0]
    YB = data[:, 1]
    x, y = generate_four_digit_NACA('2412', len(XB), 1)

    assert stats.spearmanr(YB, y).statistic > 0.99
    assert len(x) == len(XB)


def test_generate_four_digit_NACA_2():
    data = np.loadtxt('airfoils/NACA4412.txt')
    XB = data[:, 0]
    YB = data[:, 1]
    x, y = generate_four_digit_NACA('2412', len(XB), 1)

    assert stats.spearmanr(XB, x).statistic > 0.99
    assert len(y) == len(YB)
