""" Equivalence between T-Test and Statistical Model """

# Import standard packages
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as sm

# Generate normally distributed data around 'reference' + 0.2
np.random.seed(123)
reference = 5
diffs = 0.2 + np.random.randn(100)
values = reference + diffs
diffs_df = pd.DataFrame({'diffs': diffs})  # as DataFrame

# t-test
(t, pVal) = stats.ttest_1samp(values, reference)  # >> p=0.048
print('The probability that the sample mean is different' +
     f' from {reference} is {pVal:5.3f}.\n')

# Equivalent linear model
result = sm.ols(formula='diffs ~ 1', data=diffs_df).fit()
print(result.summary())
