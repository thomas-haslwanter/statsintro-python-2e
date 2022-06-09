"""  Conversion for Cochran-Test """

# author:	Thomas Haslwanter
# date:		June-2022

# Import the standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.contingency_tables import cochrans_q


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


if __name__ == '__main__':

    # Dummy data
    tasks = np.array([[0,1,1,0,1,0,0,1,0,0,0,0],
                      [1,1,1,0,0,1,0,1,1,1,1,1],
                      [0,0,1,0,0,1,0,0,0,0,0,0]])


    # Calculate with statsmodels
    df = pd.DataFrame(tasks.T, columns = ['Task1', 'Task2', 'Task3'])
    sm_results = cochrans_q(df)

    print('\nStatsmodels: --------------------------')
    print(dir(sm_results))
    print(sm_results.Q)

    # Calculate with pingouin
    df_pg = cochran_matrix_2_events(tasks)
    pg_out = pg.cochran(df_pg, dv='value', within='category', subject='subject')

    print('\nPingouin: --------------------------')
    print(pg_out)
