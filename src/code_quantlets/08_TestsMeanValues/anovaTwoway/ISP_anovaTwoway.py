""" Two-way Analysis of Variance (ANOVA)
The model is formulated using the "patsy" formula description. This is very
similar to the way models are expressed in R.
"""

# Copyright(c) 2021, Thomas Haslwanter.
# All rights reserved, under the CC BY-SA 4.0 International License

# Import standard packages
import numpy as np
import pandas as pd
import pingouin as pg

# additional packages
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def anova_interaction() -> float:
    """ ANOVA with interaction: Measurement of fetal head circumference,
    by four observers in three fetuses, from a study investigating the
    reproducibility of ultrasonic fetal head circumference data.
    
    Returns
    -------
    F_statistic
    """
    
    # Get the data
    inFile = 'altman_12_6.txt'
    data = np.genfromtxt(inFile, delimiter=',')
    
    # Bring them in DataFrame-format
    df = pd.DataFrame(data, columns=['hs', 'fetus', 'observer'])
    
    # --- >>> START stats <<< ---
    # Determine the ANOVA with interaction, either with statsmodels ...
    formula = 'hs ~ C(fetus) + C(observer) + C(fetus):C(observer)'
    lm = ols(formula, df).fit()
    sm_results = anova_lm(lm)
    
    # ... or with pingouin
    pg_results = pg.anova(dv='hs', between=['fetus', 'observer'], data=df)
    # --- >>> STOP stats <<< ---
    
    print('pingouin ---------------')
    print(pg_results.round(4))
    
    print('\nstatsmodels ------------')
    print(sm_results.round(4))

    return  sm_results['F'][0]
                              

if __name__ == '__main__':
    anova_interaction()
