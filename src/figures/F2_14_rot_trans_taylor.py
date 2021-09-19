""" Solution to Exercises 'Rotation', 'Translation', 'Taylor'
in Chapter 'Python' """

# author:   Thomas Haslwanter
# date:     Sept-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt


def translation(axis):
    # Define the original points
    p_0 = [0,0]
    p_1 = [2,1]

    # Combine them to an array
    array = np.array([p_0, p_1])
    # print(array)

    # Translate the array
    translated = array + [3,1]
    # print(translated)

    # Plot the data
    plt.sca(axis)
    plt.plot(array[:,0], array[:,1])
    plt.plot(translated[:,0], translated[:,1])
    plt.title('Translation')


def rotation(axis):
    vector = [2,1]

    plt.sca(axis)
    # Draw a green line from [0,0] to [2,1]
    plt.plot([0,vector[0]], [0, vector[1]], 'g')
    
    # Coordinate system
    plt.hlines(0, -2, 2, linestyles='dotted')
    plt.vlines(0, -2, 2, linestyles='dotted')
    
    # Make sure that the x/y dimensions are equally drawn
    cur_axis = plt.gca()
    cur_axis.set_aspect('equal')
    
    # Rotate the vector
    alpha = 25
    alpha_rad = np.deg2rad(alpha)

    R = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                  [np.sin(alpha_rad), np.cos(alpha_rad)]])

    rotated =  R @ vector

    plt.plot([0, rotated[0]], [0 ,rotated[1]], 
             label='rotated', color='r', linewidth=3)
             
    plt.title('Rotation')


def taylor(axis):
    limit = 50          # [deg]
    step_size = 0.1     # [deg]    
    
    # Calculate the data
    theta_deg = np.arange(-limit, limit, step_size)    
    theta = np.deg2rad(theta_deg)

    sin_approx = theta
    cos_approx = 1 - theta**2/2

    # Plot the data
    plt.sca(axis)
    plt.plot(theta_deg, np.sin(theta), label='sin')
    plt.plot(theta_deg, np.cos(theta), label='cos')
    plt.plot(theta_deg, sin_approx, linestyle='dashed',
              label='sin approx')
    plt.plot(theta_deg, cos_approx, linestyle='dashed',
              label='cos approx')
    plt.legend()    

    # Coordinate system
    plt.hlines(0, -limit, limit, linestyles='dotted')
    plt.vlines(0, -1, 1, linestyles='dotted')

    plt.xlabel('Angle [deg]')    
    plt.title('Taylor')
    

if __name__ == '__main__':
    
    ax1 = plt.subplot(221)
    translation(ax1)

    ax2 = plt.subplot(222)
    rotation(ax2)

    ax3 = plt.subplot(212)
    taylor(ax3)

    out_file = 'S2_rot_trans_taylor.jpg'
    plt_kws = {'quality':100}
    plt.savefig(out_file, dpi=300, pil_kwargs=plt_kws)
    print(f'Solution 2 saved to {out_file}')
    plt.show()
    
