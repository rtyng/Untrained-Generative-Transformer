import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D tensor
tensor_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Tensor:")
print(tensor_2d)
print("Shape:", tensor_2d.shape)
print("Rank:", tensor_2d.ndim)

# Create a 3D tensor
tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D Tensor:")
print(tensor_3d)
print("Shape:", tensor_3d.shape)
print("Rank:", tensor_3d.ndim)

# Accessing elements in a tensor
print("\nAccessing elements:")
print("tensor_2d[1, 2]:", tensor_2d[1, 2])
print("tensor_3d[0, 1, 0]:", tensor_3d[0, 1, 0])


# Create a 2D tensor
tensor_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Visualize the tensor as an image
plt.imshow(tensor_2d, cmap='viridis')
plt.colorbar()
plt.show()





