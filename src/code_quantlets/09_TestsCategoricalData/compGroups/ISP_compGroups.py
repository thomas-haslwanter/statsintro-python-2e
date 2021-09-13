""" Analysis of categorical data
- Analysis of one proportion
- Chi-square test
- Fisher exact test
- McNemar's test
- Cochran's Q test

"""

# author: Thomas Haslwanter, date: Sept-2021

# Import standard packages
import numpy as np
import scipy.stats as stats
import pandas as pd
import pingouin as pg

# additional packages
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q


def cochran_matrix_2_events(in_mat: np.ndarray) -> pd.DataFrame:
    """Convert a 0/1-matrix to corresponding events

    Parameters
    ----------
    in_mat : matrix, with the events for each category in row-form
    
    Returns
    -------
    df : DataFrame, with columns ['subject', 'category', 'value']

    """

    out = np.nan * np.ones((1, 3))  # Dummy-value for initiation output-matrix 

    subjects = np.arange(in_mat.shape[1])
    categories = np.arange(in_mat.shape[0])

    for ii in categories:
        new = np.column_stack( (subjects, ii*np.ones(len(subjects)),
                                 in_mat[ii,:]) )
        out = np.vstack( (out, new) )

    out = out[1:,:]     # Eliminate the dummy init-row

    df = pd.DataFrame(out, columns=['subject', 'category', 'value'])
    return df


def freq2events(data: np.ndarray) -> np.ndarray:
    """ Turn a frequency matrix into a corresponding event-matrix 
    
    Parameters
    ----------
        data : 1d or 2d frequency matrix
        
    Returns
    -------
        events : corresponding 1d or 2d event array
                 In the 1d case a vector is returned,
                 in the 2d case an n*2 matrix.
    """
    
    if len(data.shape) == 1:
        events = np.array([])
        
        for ii, value in enumerate(data):
            events = np.hstack( (events, ii*np.ones(value)) )
            
    else:
        events = np.nan * np.ones( (1,2) ) # need a 2-col dummy start-matrix

        data = np.atleast_2d(data)
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                new = np.repeat([[ii, jj]], data[ii,jj], axis=0)
                events = np.vstack((events, new))
    
        events = events[1:,:]   # strip the dummy starter
        
    return events


def oneProportion()-> np.ndarray:
    """Calculate the confidence intervals of the population, based on a
    given data sample.
    The data are taken from Altman, chapter 10.2.1.
    Suppose a general practitioner chooses a random sample of 215 women from
    the patient register for her general practice, and finds that 39 of them
    have a history of suffering from asthma. What is the confidence interval
    for the prevalence of asthma?

    Returns
    -------
    ci : Confidence interval, for testing the function
    """

    # Get the data
    numTotal = 215
    numPositive = 39

    # --- >>> START stats <<< ---
    # Calculate the confidence intervals
    p = float(numPositive)/numTotal
    se = np.sqrt(p*(1-p)/numTotal)
    td = stats.t(numTotal-1)
    ci = p + np.array([-1,1])*td.isf(0.025)*se
    # --- >>> STOP stats <<< ---

    # Print them
    print('ONE PROPORTION ----------------------------------------')
    print('The confidence interval for the given sample is ' +
          f'{ci[0]:5.3f} to {ci[1]:5.3f}')
    
    return ci


def chiSquare() -> float:
    """ Application of a chi square test to a 2x2 table.
    The calculations are done with and without Yate's continuity
    correction.
    Data are taken from Altman, Table 10.10:
    Comparison of number of hours' swimming by swimmers with or without erosion
    of dental enamel.
    >= 6h: 32 yes, 118 no
    <  6h: 17 yes, 127 no

    Returns
    -------
    chi2_corrected  : f- and p-value, for testing

    """

    # Enter the data
    obs = np.array([[32, 118],
                    [17, 127]])

    # obs = np.array([[43, 9],
    #                 [44, 4]])

    # --- >>> START stats <<< ---
    # Calculate the chi-square test
    # with statsmodels ...
    chi2_corrected = stats.chi2_contingency(obs, correction=True)
    chi2_uncorrected = stats.chi2_contingency(obs, correction=False)

    # ... and with pingouin
    events = freq2events(obs)
    df = pd.DataFrame(events, columns=['x', 'y'])
    pg_out = pg.chi2_independence(df, 'x', 'y')
    # --- >>> STOP stats <<< ---

    # Print the result
    print('\nCHI SQUARE --------------------------------------------------')
    print('The corrected chi2 value is ' +
          f'{chi2_corrected[0]:5.3f}, with p={chi2_corrected[1]:5.3f}')

    print('The uncorrected chi2 value is ' +
          f'{chi2_uncorrected[0]:5.3f}, with p={chi2_uncorrected[1]:5.3f}')

    print('\nPingouin:')
    print(pg_out)

    return chi2_corrected


def fisherExact():
    """Fisher's Exact Test:
    Data are taken from Altman, Table 10.14
    Spectacle wearing among juvenile delinquensts and non-delinquents who failed
    a vision test.

    Spectecle wearers: 1 delinquent, 5 non-delinquents
    Non-spectacle wearers: 8 delinquents, 2 non-delinquents
    """

    # Enter the data
    obs = np.array([[1,5], [8,2]])

    # --- >>> START stats <<< ---
    # Calculate the Fisher Exact Test
    # Note that by default, the option "alternative='two-sided'" is set;
    # other options are 'less' or 'greater'.
    fisher_result = stats.fisher_exact(obs)
    # --- >>> STOP stats <<< ---

    # Print the result
    print('\nFISHER --------------------------------------------------------')
    print('The probability of obtaining a distribution at least as extreme ' +
            'as the one that was actually observed, assuming that the null ' +
            f'hypothesis is true, is: {fisher_result[1]:5.3f}.')
    
    return fisher_result


def cochranQ():
    """Cochran's Q test: 12 subjects are asked to perform 3 tasks. The outcome
    of each task is "success" or "failure". The results are coded 0 for failure
    and 1 for success. In the example, subject 1 was successful in task 2, but
    failed tasks 1 and 3.
    Is there a difference between the performance on the three tasks?
    """
    
    tasks = np.array([[0,1,1,0,1,0,0,1,0,0,0,0],
                      [1,1,1,0,0,1,0,1,1,1,1,1],
                      [0,0,1,0,0,1,0,0,0,0,0,0]])
    
    # I prefer a DataFrame here, as it indicates directly what the values mean
    df = pd.DataFrame(tasks.T, columns = ['Task1', 'Task2', 'Task3'])
    
    # --- >>> START stats <<< ---
    # with statsmodels ...
    sm_results = cochrans_q(df)     #(Q, pVal)

    # ... and with pingouin
    df_pg = cochran_matrix_2_events(tasks)
    pg_out = pg.cochran(df_pg, dv='value', within='category',
                         subject='subject')
    # --- >>> STOP stats <<< ---

    print('\nStatsmodels: --------------------------')
    print(sm_results)

    print('\nPingouin: --------------------------')
    print(pg_out)

    print('\nCOCHRAN\'S Q ------------------------------------------------')
    print(f'Q = {sm_results.statistic:5.3f}, p = {sm_results.pvalue:5.3f}')
    if sm_results.pvalue < 0.05:
        print("There is a significant difference between the three tasks.")
    

def tryMcnemar():
    """McNemars Test should be run in the "exact" version, even though
    approximate formulas are
    typically given in the lecture scripts. Just ignore the statistic that is
    returned, because it is different for the two options.
    
    In the following example, a researcher attempts to determine if a drug has
    an effect on a particular disease. Counts of individuals are given in the
    table, with the diagnosis (disease: present or absent) before treatment
    given in the rows, and the diagnosis after treatment in the columns. The
    test requires the same subjects to be included in the before-and-after
    measurements (matched pairs).
    """
    
    
    f_obs = np.array([[101, 121],[59, 33]])

    # with statsmodels ....
    sm_out = mcnemar(f_obs, exact=False, correction=True)  # statistic, pvalue
    # (statistic, pVal) = mcnemar(f_obs)

    # ... and with pingouin
    events = freq2events(f_obs)
    df = pd.DataFrame(events, columns=['x', 'y'])
    pg_out = pg.chi2_mcnemar(df, 'x', 'y', correction=True)

    print('\nMCNEMAR\'S TEST ---------------------------------------------')
    print(f'p = {sm_out.pvalue:5.3f}')
    if sm_out.pvalue < 0.05:
        print("There was a significant change in the disease by the treatment.")    
    
    print('\nPingouin:')
    print(pg_out[1:])


if __name__ == '__main__':
    #oneProportion()
    chiSquare()
    #fisherExact()
    #tryMcnemar()
    #cochranQ()

