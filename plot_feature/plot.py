import torch
import matplotlib.pyplot as plt


# Define the file path to load the tensor from
# image_name = 'density_0.05_feature'
# image_name = 'density_0.1_feature'
# image_name = 'density_0.2_feature'
image_name = 'density_0.4_feature'
# image_name = "image_in"


channel = 30


file_path = image_name + '.pt'


# Load the tensor from the file
loaded_tensor = torch.load(file_path, map_location='cpu')
if 'in' not in image_name:
    loaded_tensor = loaded_tensor[channel].unsqueeze(0)
# Display the loaded tensor
print("Loaded Tensor:")
print(loaded_tensor.shape)
print("Loaded Tensor max:", loaded_tensor.max())
print("Loaded Tensor min:", loaded_tensor.min())
if 'in' not in image_name:
    loaded_tensor = (loaded_tensor - loaded_tensor.min()) / (loaded_tensor.max() - loaded_tensor.min())
# Convert the tensor to a NumPy array
numpy_array = loaded_tensor.numpy()

# Transpose the array to match the image format (32, 32, 3)
image_array = numpy_array.transpose(1, 2, 0)

# Display the image
plt.imshow(image_array)
plt.axis('off')  # Hide axis ticks and labels
plt.show()
save_path = image_name + '.png'  # Replace with the desired file path
plt.savefig(save_path)