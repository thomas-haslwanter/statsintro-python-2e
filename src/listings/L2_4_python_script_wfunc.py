""" Short demonstration of a Python script.
After a short one-line description of the content, the header can contain
further details.
"""

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from collections import tup

def get_data(amp:float = 1, freq:float=0.5, duration:float=10): 
    """ Generate a sine-wave
    
    Parameters
    ----------
    
    """
    # Generate the time-values
    t = np.arange(0, 10, 0.1)
    
    # Set the frequency, and calculate the sine-value
    freq = 0.5
    omega = 2 * np.pi * freq
    x = amp * np.sin(omega * t)
    
    return (t, x)


def plot_data(t_in, x_in):
    # Plot the data
    plt.plot(t,x)
    
    # Format the plot
    plt.xlabel('Time[sec]')
    plt.ylabel('Values')

    
def disp_and_save(out_file=None):
    # Generate a figure, one directory up, and let the user know about it
    out_file = '../Sinewave.jpg'
    plt.savefig(out_file, dpi=200, quality=90)
    print(f'Image has been saved to {out_file}')
    
    # Put it on the screen
    plt.show()

    
if __name__ == '__main__':
    t, x = get_data()
    plot_data(t, x)
    disp_and_save()
    