import concurrent.futures
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from container import Container

# kmeans 时间复杂度为O(nkt) k=3, t=5
class KmeansCalculate:
    def __init__(self, container, depth_map):
        self.max_iter = 5
        self.tol = 0.001
        self.depth_map = depth_map
        self.areas = self.calculate_areas(container.box)
        self.final_distance = 0

    def calculate_areas(self, box):
        x_min, y_min, x_max, y_max = box
        x_offset = x_max - x_min
        y_offset = y_max - y_min

        return [
            (x_min + x_offset * 0.33, y_min, x_min + x_offset * 0.66, y_max),  # Area 1
            (x_min, y_min + y_offset * 0.5, x_min + x_offset * 0.33, y_max),  # Area 2
            (x_min + x_offset * 0.66, y_min + y_offset * 0.5, x_max, y_max),  # Area 3
        ]

    def process_area(self, area):
        x1, y1, x2, y2 = map(int, area)
        cropped = self.depth_map[y1:y2, x1:x2].reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, max_iter=self.max_iter, tol=self.tol)
        labels = kmeans.fit_predict(cropped)
        centers = kmeans.cluster_centers_

        count_label = [0] * 3
        for label in labels:
            count_label[label] += 1

        total_points = len(labels)
        z_represent = sum(centers[i][0] * count_label[i] for i in range(3)) / total_points
        return z_represent

    def kmeans_classify(self):
        z_represent_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to the executor for each area
            future_to_area = {executor.submit(self.process_area, area): area for area in self.areas}
            for future in concurrent.futures.as_completed(future_to_area):
                z_represent = future.result()
                z_represent_list.append(z_represent)
        self.final_distance = sum(z_represent_list) / len(z_represent_list)


if __name__ == "__main__":
    container = Container(id=1, box=[190, 190, 200, 200])
    depth_map = np.random.randint(0, 256, (1080, 1440)).astype(np.float32)

    start_time = time.time()
    kmeansCalculate = KmeansCalculate(container, depth_map)
    kmeansCalculate.kmeans_classify()
    end_time = time.time()
    print("KMeans Classification completed in {:.2f} seconds.".format(end_time - start_time))

    print("Final distance with kmeans: ", kmeansCalculate.final_distance)
    # 显示深度图
    plt.figure(figsize=(14.4, 10.8))  # 按照比例调整图像大小
    plt.imshow(depth_map, cmap='gray')  # 使用灰度色彩映射
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()
