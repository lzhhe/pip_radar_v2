from matplotlib import pyplot as plt
import numpy as np

# Parameters
square_size = 1  # Size of each square: 1 unit
rows = 8  # Number of rows
cols = 12  # Number of columns
width = cols * square_size  # Width of the checkerboard
height = rows * square_size  # Height of the checkerboard

# Create the checkerboard pattern
checkerboard = np.zeros((rows, cols))
checkerboard[1::2, ::2] = 1
checkerboard[::2, 1::2] = 1

# Plotting
plt.figure(figsize=(12, 8))
plt.imshow(checkerboard, cmap='gray', interpolation='nearest')
plt.axis('off')  # Turn off the axis
plt.title('8x12 Calibration Pattern')
plt.show()
plt.savefig('8x12_Calibration_Pattern.png', bbox_inches='tight', pad_inches=0)