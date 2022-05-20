""" Demonstration of some pandas data handling functionality
    - grouping of data
    - pivoting
    - handling NaN's
"""

# author:   Thomas Haslwanter
# date:     Dec-2021

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


def pivoting() -> None:
    """ Demonstration of pandas pivot_table """

    # Generate some string data
    data = '''name, exam, trial, points
    Peter, midTerm, 1, 40
    Paul, midTerm, 1, 60
    Mary, midTerm, 1, 20
    Mary, midTerm, 2, 70
    Peter, final, 1, 60
    Paul, final, 1, 20
    Mary, final, 1, 80
    Paul, final, 2, 75
    '''

    # Write them to a buffer
    buffer = StringIO()     # creating an empty buffer
    buffer.write(data)

    # Read it from the buffer into a pandas DataFrame
    buffer.seek(0)
    df = pd.read_csv(buffer, sep='[, ]+', engine='python')

    # Generate a pivot table
    pd.pivot_table(df, index=['exam', 'name'], values=['points'],
            columns=['trial'])
    out =  pd.pivot_table(df, index=['exam', 'name'], values=['points'],
            aggfunc=[np.max, len])
    print(out)


def grouping() -> None:
    """ Demonstration of pandas grouping function """

    # Generate some data
    data = pd.DataFrame({
            'Gender': ['f', 'f', 'm', 'f', 'm', 'm', 'f', 'm', 'f', 'm', 'm'],
            'TV': [3.4, 3.5, 2.6, 4.7, 4.1, 4.1, 5.1, 3.9, 3.7, 2.1, 4.3] })

    # Group the data
    grouped = data.groupby('Gender')

    # Do some overview statistics
    print(grouped.describe())

    # Grouped data can also be plotted
    grouped.boxplot()
    plt.show()

    # Get the groups as DataFrames
    df_female = grouped.get_group('f')


def handle_nans() -> None:
    """ Show some of the options of handling nan-s in Pandas """

    print('--- Handling nan-s in Pandas ---')

    # Generate data containing "nan":
    x = np.arange(7, dtype=float)
    y = x**2

    x[3] = np.nan
    y[ [2,5] ] = np.nan

    # Put them in a Pandas DataFrame
    df = pd.DataFrame({'x':x, 'y':y})
    print(df)

    # Different ways of handling the "nan"s in a DataFrame:

    print('Drop all lines containint nan-s:')
    print(df.dropna())     # Drop all rows containing nan-s

    print('Replaced with the next-lower value:')
    print(df.fillna(method='pad'))  # Replace with the next-lower value

    print('Replaced with an interpolated value:')
    print(df.interpolate())         # Replace with an interpolated value


if __name__ == '__main__':
    print('\n', '-'*60)
    grouping()

    print('\n', '-'*60)
    pivoting()

    print('\n', '-'*60)
    handle_nans()

    """
    This produces the following printout:

     ------------------------------------------------------------
              TV
           count      mean       std  min    25%  50%  75%  max
    Gender
    f        5.0  4.080000  0.769415  3.4  3.500  3.7  4.7  5.1
    m        6.0  3.516667  0.926103  2.1  2.925  4.0  4.1  4.3

     ------------------------------------------------------------
                    amax    len
                  points points
    exam    name
    final   Mary      80      1
            Paul      75      2
            Peter     60      1
    midTerm Mary      70      2
            Paul      60      1
            Peter     40      1

     ------------------------------------------------------------
        --- Handling nan-s in Pandas ---
         x     y
    0  0.0   0.0
    1  1.0   1.0
    2  2.0   NaN
    3  NaN   9.0
    4  4.0  16.0
    5  5.0   NaN
    6  6.0  36.0
    Drop all lines containint nan-s:
         x     y
    0  0.0   0.0
    1  1.0   1.0
    4  4.0  16.0
    6  6.0  36.0
    Replaced with the next-lower value:
         x     y
    0  0.0   0.0
    1  1.0   1.0
    2  2.0   1.0
    3  2.0   9.0
    4  4.0  16.0
    5  5.0  16.0
    6  6.0  36.0
    Replaced with an interpolated value:
         x     y
    0  0.0   0.0
    1  1.0   1.0
    2  2.0   5.0
    3  3.0   9.0
    4  4.0  16.0
    5  5.0  26.0
    6  6.0  36.0
    """
