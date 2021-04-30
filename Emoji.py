import matplotlib.pyplot as plt
from scipy import spatial
from PIL import Image
import numpy as np

# Reading images and resizing them
G_sm = np.array(Image.open('fans.png').resize([60, 60]).getdata()).reshape([60, 60, 3])/256

# Show images
plt.figure()
plt.imshow(G_sm)
plt.title('Original Image')
plt.show()


# Read emoji data
emoji_array = np.load("emojis_16.npy")

# Get the average colour value of an emoji
emoji_mean_array = np.array([ar.mean(axis=(0,1)) for ar in emoji_array])

# Store the average colour value of each emoji in the tree to speed up the search
tree = spatial.KDTree(emoji_mean_array)


indices = []
# Flat integer array, one-dimensional
flattened_img = G_sm.reshape(-1, G_sm.shape[-1])
print(flattened_img.shape)

# Pixels that match the most similar emoji
for pixel in flattened_img:
    pixel_ = np.concatenate((pixel, [1]))
    # Querying the most recent index
    _, index = tree.query(pixel_)
    indices.append(index)


# Get the corresponding emoji from the index
emoji_matches = emoji_array[indices]

# Get the height of the image
dim = G_sm.shape[0]
print(dim)

# Set the size of the final generated image, the shape of each emoji is (16,16,4), R, G, B, alpha
resized_ar = emoji_matches.reshape((dim, dim, 16, 16,4))


# Convert single emoji patch (5D)
# Using numpy to generate a complete image (3D)
final_img = np.block([[[x] for x in row] for row in resized_ar])


# Setting the canvas
plt.figure()
# Removal of axes
plt.axis('off')
# Show images
plt.imshow(final_img)
# Save emoji mosaic style images with white borders removed
plt.savefig('image_emoji.png', bbox_inches="tight", pad_inches=0.0)

plt.show()