""" Solution to Exercise 'Translation', the Chapter 'Python' """

# author:   Thomas Haslwanter
# date:     April-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt

# Define the original points
p_0 = [0,0]
p_1 = [2,1]

# Combine them to an array
array = np.array([p_0, p_1])
print(array)

# Translate the array
translated = array + [3,1]
print(translated)

# Plot the data
plt.plot(array[:,0], array[:,1], label='original')
plt.plot(translated[:,0], translated[:,1], label='translated')

# Format and show the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


""" Solution to Exercise 'Rotation', Chapter 'Python' """

# author:   Thomas Haslwanter
# date:     April-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt


def rotate_me(in_vector:np.ndarray, alpha:float) -> np.ndarray:
    """Function that rotates a vector in 2 dimensions
    
    Parameters
    ----------
    in_vector : vector (2,) or array (:,2)
                vector(s) to be rotated
    alpha : rotation angle [deg]
    
    Returns
    -------
    rotated_vector : vector (2,) or array (:,2)
                rotated vector
    
    Examples
    --------
    perpendicular = rotate_me([1,2], 90)
    
    """
    
    alpha_rad = np.deg2rad(alpha)
    R = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
    [np.sin(alpha_rad), np.cos(alpha_rad)]])
    return R @ in_vector


if __name__ == '__main__':
    
    vector = [2,1]
    # Draw a green line from [0,0] to [2,1]
    plt.plot([0,vector[0]], [0, vector[1]], 'g', label='original')
    
    # Coordinate system
    plt.hlines(0, -2, 2, linestyles='dashed')
    plt.vlines(0, -2, 2, linestyles='dashed')
    
    # Make sure that the x/y dimensions are equally drawn
    cur_axis = plt.gca()
    cur_axis.set_aspect('equal')
    
    # Rotate the vector
    rotated = rotate_me(vector, 25)
    plt.plot([0, rotated[0]], [0 ,rotated[1]], 
             label='rotated', 
             color='r', 
             linewidth=3)
             
    plt.legend()
    plt.show()


""" Solution to Exercise 'Taylor', Chapter 'Python' """

# author:   Thomas Haslwanter
# date:     April-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def approximate(angle:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function that calculates a second order approximation to sine and cosine
    
    Parameters
    ----------
    angle : angle [deg]
    
    Returns
    -------
    approx_sine :  approximated sine 
    approx_cosine :  approximated cosine 
    
    Examples
    --------
    alpha = 0.1
    sin_ax, cos_ax = approximate(alpha)

    Notes
    -----
    Input can also be a single float
    
    """
    
    sin_approx = angle
    cos_approx = 1 - angle**2/2
    
    return (sin_approx, cos_approx)


if __name__ == '__main__':
    limit = 50          # [deg]
    step_size = 0.1     # [deg]    
    
    # Calculate the data
    theta_deg = np.arange(-limit, limit, step_size)    
    theta = np.deg2rad(theta_deg)
    sin_approx, cos_approx = approximate(theta)
    
    # Plot the data
    plt.plot(theta_deg, np.column_stack((np.sin(theta), np.cos(theta))), label='exact')    
    plt.plot(theta_deg, np.column_stack((sin_approx, cos_approx)), 
             linestyle='dashed', 
             label='approximated')             
    plt.legend()    
    plt.xlabel('Angle [deg]')    
    plt.title('sine and cosine')
    out_file = 'approximations.png'
    plt.savefig(out_file, dpi=200)
    print(f'Resulting image saved to {out_file}')
    
    plt.show()
