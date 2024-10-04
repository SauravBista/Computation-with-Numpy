# Introduction
# In this notebook we'll learn how to use NumPy to work with numerical data.

# Import Statements
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc  # contains an image of a racoon!
from PIL import Image  # for reading image files

# Understanding NumPy's ndarray
# NumPy's most amazing feature is the powerful ndarray.

# 1-Dimensional Arrays (Vectors)
my_array = np.array([1.1, 9.2, 8.1, 4.7])
my_array.ndim  # Output: 1

# 2-Dimensional Arrays (Matrices)
array_2d = np.array([[1, 2, 3, 9],
                      [5, 6, 7, 8]])
array_2d[:, 0]  # Output: array([1, 5])

# N-Dimensional Arrays (Tensors)
# Point:
# How many dimensions does the array below have?
# What is its shape (i.e., how many elements are along each axis)?
# Try to access the value 18 in the last line of code.
# Try to retrieve a 1 dimensional vector with the values [97, 0, 27, 18]
# Try to retrieve a (3,2) matrix with the values [[ 0, 4], [ 7, 5], [ 5, 97]]
# Hint: You can use the : operator just as with Python Lists.

mystery_array = np.array([[[0, 1, 2, 3],
                            [4, 5, 6, 7]],
                           [[7, 86, 6, 98],
                            [5, 1, 0, 4]],
                           [[5, 36, 32, 48],
                            [97, 0, 27, 18]]])

# Note all the square brackets!
mystery_array.ndim  # Output: 3
mystery_array.shape  # Output: (3, 2, 4)
mystery_array[2, 1]  # Output: array([97,  0, 27, 18])
mystery_array[:, :, 0]  # Output: array([[ 0,  4],
#                                      [ 7,  5],
#                                      [ 5, 97]])

# NumPy Mini-Points
# Point 1: Use .arange() to create a vector a with values ranging from 10 to 29. You should get this:
# print(a)
# [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]

a = np.arange(10, 30)
print(a)  # Output: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]

# Point 2: Use Python slicing techniques on a to:
# Create an array containing only the last 3 values of a
# Create a subset with only the 4th, 5th, and 6th values
# Create a subset of a containing all the values except for the first 12 (i.e., [22, 23, 24, 25, 26, 27, 28, 29])
# Create a subset that only contains the even numbers (i.e, every second number)

b = a[-3:]  # Output: [27 28 29]
print(b)
print(a[3:6])  # Output: [13 14 15]
print(a[12:])  # Output: [22 23 24 25 26 27 28 29]
print(a[::2])  # Output: [10 12 14 16 18 20 22 24 26 28]

# Point 3: Reverse the order of the values in a, so that the first element comes last:
# [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
print(a[::-1])  # Output: [29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10]

# Point 4: Print out all the indices of the non-zero elements in this array: [6,0,9,0,0,5,0]
b = np.array([6, 0, 9, 0, 0, 5, 0])
nz_indices = np.nonzero(b)
print(nz_indices)  # Output: (array([0, 2, 5]),)

# Point 5: Use NumPy to generate a 3x3x3 array with random numbers
# Hint: Use the .random() function
from numpy.random import random
z = random((3, 3, 3))
print(z)

# Point 6: Use .linspace() to create a vector x of size 9 with values spaced out evenly between 0 to 100 (both included).
x = np.linspace(0, 100, num=9)
print(x)  # Output: [  0.   12.5  25.   37.5  50.   62.5  75.   87.5 100. ]

# Point 7: Use .linspace() to create another vector y of size 9 with values between -3 to 3 (both included). Then plot x and y on a line chart using Matplotlib.
y = np.linspace(-3, 3, num=9)
print(y)
plt.plot(x, y)
plt.show()

# Point 8: Use NumPy to generate an array called noise with shape 128x128x3 that has random values. Then use Matplotlib's .imshow() to display the array as an image.
noise = random((128, 128, 3))
plt.imshow(noise)

# Linear Algebra with Vectors
v1 = np.array([4, 5, 2, 7])
v2 = np.array([2, 1, 3, 3])

# Python Lists vs ndarrays
list1 = [4, 5, 2, 7]
list2 = [2, 1, 3, 3]

# Matrix Multiplication with @ and .matmul()
a1 = np.array([[1, 3],
               [0, 1],
               [6, 2],
               [9, 7]])

b1 = np.array([[4, 1, 3],
               [5, 8, 5]])

print(f'{a1.shape}: a has {a1.shape[0]} rows and {a1.shape[1]} columns.')
print(f'{b1.shape}: b has {b1.shape[0]} rows and {b1.shape[1]} columns.')
print('Dimensions of result: (4x2)*(2x3)=(4x3)')
# Output: (4, 2): a has 4 rows and 2 columns.
#         (2, 3): b has 2 rows and 3 columns.
#         Dimensions of result: (4x2)*(2x3)=(4x3)
ans = np.matmul(a1, b1)
print(ans)  # Output: [[19 25 18]
#                          [ 5  8  5]
#                          [34 22 28]
#                          [71 65 62]]

# Manipulating Images as ndarrays
img = misc.face()
plt.imshow(img)

# Point: What is the data type of img? Also, what is the shape of img and how many dimensions does it have? What is the resolution of the image?
print(type(img))  # Output: numpy.ndarray
print(img.shape)  # Output: (768, 1024, 3)
print(img.ndim)  # Output: 3

# Point: Convert the image to black and white. The values in our img range from 0 to 255.
# Divide all the values by 255 to convert them to sRGB, where all the values are between 0 and 1.
# Next, multiply the sRGB array by the grey_vals to convert the image to grey scale.
# Finally use Matplotlib's .imshow() together with the colormap parameter set to gray cmap=gray to look at the results.
grey_vals = np.array([0.2126, 0.7152, 0.0722])
srgb = img / 255
bw = np.matmul(srgb, grey_vals)
plt.imshow(bw, cmap='gray')

# Point: Can you manipulate the images by doing some operations on the underlying ndarrays? See if you can change the values in the ndarray so that:
# 1) You flip the grayscale image upside down
# 2) Rotate the colour image
# 3) Invert (i.e., solarize) the colour image. To do this you need to convert all the values in the img array to 255 minus the current value in that position.

bw_flipped = np.flip(bw, axis=0)
plt.imshow(bw_flipped, cmap='gray')

# Rotate the image
rotated_img = np.rot90(img)
plt.imshow(rotated_img)

# Invert the colour image
solarized_img = 255 - img
plt.imshow(solarized_img)

# Closing Comments
# By the end of this notebook, you should now know how to use NumPy to work with numerical data.
