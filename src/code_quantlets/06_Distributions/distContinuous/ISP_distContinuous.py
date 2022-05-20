""" Different continuous distribution functions.
- T-distribution
- F-distribution
- Chi2-distribution
- Exponential
- Weibull
"""

# author: Thomas Haslwanter, date: March-2022

# Import standard packages
# note: "# type: ignore" eliminates some warnings in the static typechecking
import numpy as np
import matplotlib.pyplot as plt     # type: ignore
from scipy import stats             # type: ignore
import seaborn as sns               # type: ignore

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..',  '..', 'Utilities'))
try:
    from ISP_mystyle import setFonts, showData

except ImportError:
    # Ensure correct performance otherwise
    def setFonts(*options):
        return
    def showData(*options):
        plt.show()
        return


sns.set(context='notebook', style='ticks', palette='muted', font_scale=1.5)


def show_t():
    """Utility function to show T distributions"""

    t = np.arange(-5, 5, 0.05)
    t_vals = [1, 5]

    normal = stats.norm.pdf(t)
    t1 = stats.t.pdf(t, t_vals[0])
    t5 = stats.t.pdf(t, t_vals[1])

    plt.plot(t, normal, '--',  label='normal')
    plt.plot(t, t1, label='df=1')
    plt.plot(t, t5, label='df=5')
    plt.legend()

    plt.xlim(-5, 5)
    plt.xlabel('X')
    plt.ylabel('pdf(X)')

    out_file = 'dist_t.jpg'
    showData(out_file)


def show_chi2():
    """Utility function to show Chi2 distributions"""

    t = np.arange(0, 8, 0.05)
    Chi2Vals = [1, 2, 3, 5]

    for chi2 in Chi2Vals:
        plt.plot(t, stats.chi2.pdf(t, chi2), label='n={0}'.format(chi2))
    plt.legend()

    # plt.xlim(0, 8)
    plt.xlabel('X')
    plt.ylabel('pdf(X)')
    plt.tight_layout()

    out_file = 'dist_chi2.jpg'
    showData(out_file)


def show_f():
    """Utility function to show F distributions"""

    t = np.arange(0, 3, 0.01)
    d1s = [1, 2, 5, 100]
    d2s = [1, 1, 2, 100]

    for (d1,d2) in zip(d1s,d2s):
        plt.plot(t, stats.f.pdf(t, d1, d2), label='F({0}/{1})'.format(d1,d2))
    plt.legend()

    plt.xlim(0, 3)
    plt.xlabel('X')
    plt.ylabel('pdf(X)')
    plt.tight_layout()
    plt.legend()

    out_file = 'dist_f.jpg'
    showData(out_file)


def show_exp():
    """Utility function to show exponential distributions"""

    t = np.arange(0, 3, 0.01)
    lambdas = [0.5, 1, 1.5]

    for par in lambdas:
        plt.plot(t, stats.expon.pdf(t, 0, par),
                label='$\lambda={0:3.1f}$'.format(par))
    plt.legend()

    plt.xlim(0, 3)
    plt.xlabel('X')
    plt.ylabel('pdf(X)')
    plt.tight_layout()
    plt.legend()

    out_file = 'dist_exp.jpg'
    showData(out_file)


def show_uniform():
    """Utility function to show uniform distributions"""

    x = np.arange(0, 10, 0.1)
    a, b = 4, 3

    ud = stats.uniform(a, b)
    plt.plot(x, ud.pdf(x))

    plt.xlabel('X')
    plt.ylabel('pdf(X)')
    plt.tight_layout()

    out_file = 'dist_uniform.jpg'
    showData(out_file, margin=0.02)


def show_weibull():
    """Utility function to show Weibull distributions"""

    t = np.arange(0, 2.5, 0.01)
    ks = [0.5, 1, 1.5, 5]

    for k in ks:
        wd = stats.weibull_min(k)
        plt.plot(t, wd.pdf(t), label='k = {0:.1f}'.format(k))

    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    plt.xlabel('X')
    plt.ylabel('pdf(X)')
    plt.legend()

    out_file = 'Weibull_PDF.jpg'
    showData(out_file)


if __name__ == '__main__':
    show_t()
    show_chi2()
    show_f()
    show_exp()
    show_weibull()
    show_uniform()
