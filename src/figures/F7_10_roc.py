""" Show the origin of ROC-curves
ROC curves plot "sensitivity" against "1-specificity".
The example here uses two normally distributed groups, with a mean of 1 and 6,
respectively, and a standard deviation of 2.
"""

# author: Thomas Haslwanter, date: Dec-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..', 'Code_Quantlets', 'Utilities'))
try:
    from ISP_mystyle import showData, setFonts

except ImportError:
# Ensure correct performance otherwise
    def showData(*options):
        plt.show()
        return


def arrow_bidir(ax, start, end, headWidth=0.01):
    """Plot a bidirectional arrow"""

       # For the arrow, find the start

    start = np.array(start)
    end = np.array(end)
    delta = end - start

    ax.arrow(start[0], start[1], delta[0], delta[1],
              width=headWidth, length_includes_head=True,
              head_length=headWidth*3, head_width=headWidth*5, color='#BBBBBB')

    ax.arrow(end[0], end[1], -delta[0], -delta[1],
              width=headWidth, length_includes_head=True,
              head_length=headWidth*3, head_width=headWidth*5, color='#BBBBBB')


def main():
    # Calculate the PDF-curves
    x = np.linspace(-10, 15, 201)
    normals = stats.norm(1,2)
    patients = stats.norm(6,2)
    y1 = normals.pdf(x)
    y2 = patients.pdf(x)

    xtick_locations = [-10, -3, 1, 3.5, 6, 9, 15]
    xtick_labels = ['0', '1', '2', '3', '4', '5', '6']

    # Axes locations
    ROC = {'left': 0.35,
           'width': 0.36,
           'bottom': 0.1,
           'height': 0.47}

    PDF = {'left': 0.1,
           'width': 0.8,
           'bottom': 0.65,
           'height': 0.3}

    ROC = {'left': 0.30,
           'width': 0.30,
           'bottom': 0.1,
           'height': 0.4}

    PDF = {'left': 0.1,
           'width': 0.8,
           'bottom': 0.65,
           'height': 0.3}

    rect_ROC = [ROC['left'], ROC['bottom'], ROC['width'], ROC['height']]
    rect_PDF = [PDF['left'], PDF['bottom'], PDF['width'], PDF['height']]

    fig = plt.figure(figsize=(12,8))
    setFonts(18)

    ax1 = plt.axes(rect_PDF)
    ax2 = plt.axes(rect_ROC)

    # Plot and label the PDF-curves
    ax1.plot(x,y1, label='Normals')
    ax1.fill_between(x,0,y1, where=x>3, facecolor='C0', alpha=0.3)
    ax1.annotate('False\nPositive Rate',
                 xy=(3.2, 0.09),
                 xytext=(0.8, 0.11),
                 fontsize=14,
                 horizontalalignment='center',
                 arrowprops=dict(facecolor='C0',
                     lw=0.5))

    ax1.plot(x,y2, label='Patients')
    ax1.fill_between(x,0,y2, where=x>3, facecolor='C1', alpha=0.2)

    ax1.set_xticks(xtick_locations)
    ax1.set_xticklabels(xtick_labels, color='C2')
    ax1.set_xlim(-10.3, 15.3)

    ax1.annotate('True\nPositive Rate',
                 xy=(7, 0.05),
                 xytext=(11,0.07),
                 fontsize=14,
                 horizontalalignment='center',
                 arrowprops=dict(facecolor='C1',
                     lw=0.5))

    ax1.annotate('Threshold',
                 xy=(3.18, 0.00),
                 xytext=(3.18,-0.06),
                 fontsize=18,
                 horizontalalignment='center',
                 arrowprops=dict(facecolor='k',
                     arrowstyle='-|>'))

    ax1.text(5.5, -0.045, 'sick', fontsize=16, style='italic')
    ax1.text(-1, -0.045, 'healthy', fontsize=16, style='italic')

    ax1.set_ylabel('PDF')
    ax1.set_ylabel('PDF')
    ax1.set_ylim([0, 0.21])
    ax1.legend()
    ax1.set_yticklabels([])
    plt.tight_layout()

    # Plot the ROC-curve
    ax2.plot(normals.sf(x), patients.sf(x), 'k')
    ax2.plot(np.array([0,1]), np.array([0,1]), 'k--')

    x_values = normals.sf(xtick_locations)
    y_values = patients.sf(xtick_locations)

    ax2.plot(x_values, y_values, marker='o', color='C2', lw=0)

    for x,y,label in zip(x_values, y_values, xtick_labels):
        ax2.annotate(label, xy=(x,y), textcoords='data',
                     horizontalalignment='left',
                     verticalalignment='top',
                     color='C2')

    # Format the ROC-curve
    ax2.set_title('ROC-Curve')
    ax2.axis('square')
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')

    arrow_bidir(ax2, (0.5,0.5), (0.095, 0.885))

    # Show the plot, and create a figure
    showData('ROC.jpg')

if __name__ == '__main__':
    main()
