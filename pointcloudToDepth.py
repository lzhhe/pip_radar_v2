import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_car_point_cloud(num_points=1000):
    length, width, height = 4, 2, 1.5

    points = np.random.rand(num_points, 3)
    points[:, 0] = points[:, 0] * length - length / 2  # X坐标
    points[:, 1] = points[:, 1] * width - width / 2    # Y坐标
    points[:, 2] = points[:, 2] * height               # Z坐标

    return points

# 生成点云
car_points = generate_car_point_cloud()

# 绘制点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(car_points[:, 0], car_points[:, 1], car_points[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Car Point Cloud')
plt.show()
