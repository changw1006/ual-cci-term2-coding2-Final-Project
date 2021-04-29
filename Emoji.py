import matplotlib.pyplot as plt
from scipy import spatial
from PIL import Image
import numpy as np

# 读取图片，并且修改图片大小
G_sm = np.array(Image.open('fans.png').resize([60, 60]).getdata()).reshape([60, 60, 3])/256

# 显示图片
plt.figure()
plt.imshow(G_sm)
plt.title('Original Image')
plt.show()


# 读取emoji数据
emoji_array = np.load("emojis_16.npy")

# 获取emoji的平均颜色值
emoji_mean_array = np.array([ar.mean(axis=(0,1)) for ar in emoji_array])

# 将得到的每个emoji平均颜色值存储在树中以加快搜索速度
tree = spatial.KDTree(emoji_mean_array)


indices = []
# 平整数组，一维
flattened_img = G_sm.reshape(-1, G_sm.shape[-1])
print(flattened_img.shape)

# 匹配最相似的表情符号的像素
for pixel in flattened_img:
    pixel_ = np.concatenate((pixel, [1]))
    # 查询最近的索引
    _, index = tree.query(pixel_)
    indices.append(index)


# 从索引中获取对应的表情符号
emoji_matches = emoji_array[indices]

# 获取图片的高度
dim = G_sm.shape[0]
print(dim)

# 设置最终生成图像的大小，每个表情符号的形状都是(16,16,4)，R, G, B, alpha
resized_ar = emoji_matches.reshape((dim, dim, 16, 16,4))


# 转换单个表情符号补丁（5维）
# 使用numpy块生成完整的图像(三维)
final_img = np.block([[[x] for x in row] for row in resized_ar])


# 设置画布
plt.figure()
# 去除坐标轴
plt.axis('off')
# 显示图片
plt.imshow(final_img)
# 保存emoji马赛克风格图像，去除白边
plt.savefig('image_emoji.png', bbox_inches="tight", pad_inches=0.0)

plt.show()


