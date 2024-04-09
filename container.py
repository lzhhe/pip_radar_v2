import cv2
import numpy as np
from transformLocation import *

camera_matrix = np.array([[2.15768421e+03, 0.00000000e+00, 1.11801169e+02],
                          [0.00000000e+00, 2.15942905e+03, 2.47361876e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([[-3.99667243e-01, 9.24536500e+00, 2.26975676e-02, -2.89065215e-02, -3.42883144e+01]])

# 单位:m
# 此处只是简单的计算，长度不畸变，但是取了对角线，因为实际的检测框是要比实际大的，这样可以减一下误差# 高度也没有进行换算，但是应该有个一个3m的换算到时候再说
armor_sizes = {
    "armor1": np.array([  # 英雄
        [-0.55, -0.4, 0],  # 左上角
        [0.55, -0.4, 0],  # 右上角
        [0.55, 0.4, 0],  # 右下角
        [-0.55, 0.4, 0]  # 左下角
    ]),
    "armor2": np.array([  # 工程
        [-0.42, -0.3, 0],  # 左上角
        [0.42, -0.3, 0],  # 右上角
        [0.42, 0.3, 0],  # 右下角
        [-0.42, 0.3, 0]  # 左下角
    ]),
    "armor3": np.array([  # 步兵
        [-0.42, -0.25, 0],  # 左上角
        [0.42, -0.25, 0],  # 右上角
        [0.42, 0.25, 0],  # 右下角
        [-0.42, 0.25, 0]  # 左下角
    ]),
    "armor4": np.array([  # 步兵
        [-0.42, -0.25, 0],  # 左上角
        [0.42, -0.25, 0],  # 右上角
        [0.42, 0.25, 0],  # 右下角
        [-0.42, 0.25, 0]  # 左下角
    ]),
    "armor5": np.array([  # 步兵
        [-0.42, -0.25, 0],  # 左上角
        [0.42, -0.25, 0],  # 右上角
        [0.42, 0.25, 0],  # 右下角
        [-0.42, 0.25, 0]  # 左下角
    ]),
    "armor6": np.array([  # 哨兵
        [-0.5, -0.25, 0],  # 左上角
        [0.5, -0.25, 0],  # 右上角
        [0.5, 0.25, 0],  # 右下角
        [-0.5, 0.25, 0]  # 左下角
    ])

}

armor_weightC = {
    "armor1": 0.4,
    "armor2": 0.4,
    "armor3": 0.4,
    "armor4": 0.4,
    "armor5": 0.4,
    "armor6": 0.4,
}

armor_weightI = {
    "armor1": 0.6,
    "armor2": 0.6,
    "armor3": 0.6,
    "armor4": 0.6,
    "armor5": 0.6,
    "armor6": 0.6,
}

armor_flag_limit = {
    "armor1": 50,
    "armor2": 50,
    "armor3": 30,
    "armor4": 30,
    "armor5": 30,
    "armor6": 30,
}

robotIdDict = {  # 权重识别的哨兵的名字的armor6，这里需要转换一下
    "armor1red": 1,
    "armor2red": 2,
    "armor3red": 3,
    "armor4red": 4,
    "armor5red": 5,
    "armor6red": 7,

    "armor1blue": 101,
    "armor2blue": 102,
    "armor3blue": 103,
    "armor4blue": 104,
    "armor5blue": 105,
    "armor6blue": 107,
}

# 红方为左上
# 红方机场作为原点
# 雷达坐标为(-162.8,-592.3) 单位cm
# 相机角度暂时定为平行于围墙平行于地面
# 雷达中心为官方给出的点的中心作为计算中心
# 台子高2.5m 围栏高1.1m，因此雷达高度暂定为4m，需要一个1.5m的支架
# 地面z轴位0
# 高台高度待定
# distance 单位为m
import math

imgSize = [1503, 829]  # 测试图片尺寸
frameSize = [1440, 1080]  # 需要海康调试
radarPosition = [-162.8, 592.3, 400]  # 单位cm
# 角度需要根据ROI确定
fov_horizontal = 120  # 相机水平视场角，单位：度
fov_vertical = 120  # 相机垂直视场角，单位：度

actual_width = 2800  # 单位：cm
actual_height = 1500  # 单位：cm

# 比例因子
scale_x = imgSize[0] / actual_width
scale_y = imgSize[1] / actual_height


def calculateDistance(object_points, image_points, camera_matrix, dist_coeffs):
    """
        使用PnP算法计算从相机到物体的距离。

            参数:
            - object_points: 物体上的点的三维坐标，形状为(N, 3)。
            - image_points: 对应于object_points的点在图像上的二维坐标，形状为(N, 2)。
            - camera_matrix: 相机的内参矩阵。
            - dist_coeffs: 相机的畸变系数。

            返回:
            - distance: 物体到相机的距离。
            """
    _, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    distance = np.linalg.norm(translation_vector)
    return distance


def calculate_center_distance(box1, box2):
    # 计算第一个检测框的中心点坐标
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2

    # 计算第二个检测框的中心点坐标
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2

    # 计算两个中心点之间的距离
    distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5

    return distance


def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def calculate_score(IOU, conf, weightC, weightI):
    return IOU * weightI + conf * weightC


class Container:

    def __init__(self, id, box):
        self.id = id
        self.box = box  # car的锚框 四个点

        # 一些参数需要根据id属性进行再次分配
        self.flagLimit = 30  # 默认30 避免锁死

        self.label = ""  # 最终标签
        self.robotId = ""
        self.score = 0  # 最终评分
        self.weightI = 0.5  # 初次需要有个权重
        self.weightC = 0.5
        self.size = np.array([
            [0, 0, 0],  # 左上角
            [0, 0, 0],  # 右上角
            [0, 0, 0],  # 右下角
            [0, 0, 0]  # 左下角
        ])

        self.flag = 0  # 正确帧数标志位

        self.distance = 0
        self.disOffset = 0  # 距离矫正偏移值 这部分以后可能有用

        self.xLocation = 0  # 世界坐标系位置
        self.yLocation = 0

    def resetFlag(self):
        self.flag = 0

    def updateConfig(self):
        # print("更新配置")
        label_name = self.label[:6]
        # print("label_name: ", label_name)
        self.weightC = armor_weightC.get(label_name)
        self.weightI = armor_weightI.get(label_name)
        self.size = armor_sizes.get(label_name)
        self.flagLimit = armor_flag_limit.get(label_name)

        self.robotId = robotIdDict.get(self.label)

    def updateLabel(self, armorDict):
        if self.label is None or self.flag > self.flagLimit or self.flag == 0:  # 未分类/没有到限制不重置

            tempLabel = ""
            tempScore = 0
            for id, armor in armorDict.items():
                armorBox = armor[0]
                armorLabel = armor[1]
                armorConf = armor[2]
                # 计算两个检测框中心点之间的距离
                center_distance = calculate_center_distance(self.box, armorBox)
                if center_distance > np.sqrt(2) * max(self.box[2], self.box[3]):
                    # print("distance is far from the container box")
                    continue

                IOU = calculate_iou(self.box, armorBox)
                if IOU < 0.5:
                    continue
                else:
                    result_score = calculate_score(IOU, armorConf, self.weightC, self.weightI)
                    # print("result: ", result_score)
                    if tempScore < result_score:
                        tempLabel = armorLabel
                        tempScore = result_score

            # 华南的方案这里增加了等级机制，但是个人觉得不一定时间越长就不用重置分类，所以不加了
            if tempLabel == "":  # 避免更新帧没有更新成功
                print("can not update, no match armor")
                self.flag = self.flag + 1
                self.updateDistance()
                # 保持原有
                return
            self.flag = 1
            self.score = tempScore
            self.label = tempLabel
            self.updateConfig()
            self.updateDistance()

        else:
            self.flag = self.flag + 1
            self.updateDistance()
            # print("没有到限制不重置")
            return

    def updateDistance(self):
        if self.label and self.label[:6] in armor_sizes:

            image_points = np.array([
                [self.box[0], self.box[1]],  # 左上角
                [self.box[2], self.box[1]],  # 右上角
                [self.box[2], self.box[3]],  # 右下角
                [self.box[0], self.box[3]]  # 左下角
            ], dtype=np.float32)  # 确保为浮点数类型
            self.distance = calculateDistance(self.size, image_points, camera_matrix, dist_coeffs)
        else:
            print("Invalid label or label not found in armor_sizes. Distance not updated.")

    def print_info(self):
        print(f"ID: {self.id}")
        print(f"Box: {self.box}")
        print(f"Label: {self.label}")
        print(f"Score: {self.score}")
        print(f"WeightC: {self.weightC}")
        print(f"WeightI: {self.weightI}")
        print(f"Size: {self.size}")
        print(f"Flag: {self.flag}")
        print(f"Flag Limit: {self.flagLimit}")
        print(f"Distance: {self.distance:.2f} M" + "\n")

    def print_simple_info(self):
        print(f"ID: {self.id}")
        print(f"Box: {self.box}")
        print(f"Label: {self.label}")
        print(f"Flag: {self.flag}")
        print(f"Distance: {self.distance:.2f} units" + "\n")

    def showLocation(self):
        return getTargetImageCoordinates(self.box, self.distance)

    def showRealLocation(self):
        return calculate2DPosition(self.box, self.distance)

    # yaw 偏航角 pitch俯仰角
    def calculateYawPitch(self):
        x_min, y_min, x_max, y_max = self.box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        x_image_center = frameSize[0] / 2
        y_image_center = frameSize[1] / 2

        # 计算偏航角
        yaw = ((x_center - x_image_center) / frameSize[0]) * fov_horizontal

        # 计算俯仰角
        pitch = ((y_center - y_image_center) / frameSize[1]) * fov_vertical

        return yaw, pitch

    def calculate2DPosition(self):
        # pitch暂时没用因为已经有了高度不需要用pitch
        # 但是如果计算高台可能需要
        # 目前看起来在己方高地上 10m以内误差绝对水平（地面）误差在0.6m左右，大致可以接受，因此高台转换优先级不高
        yaw, pitch = calculateYawPitch(self.box)

        # 将角度转换为弧度
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)

        d_h = max((self.distance * 100) ** 2 - radarPosition[2] ** 2, 0) ** 0.5  # 暂时无法处理距离小于4m的
        # 同时想到可以用这个距离的角度算车是不是在高地上

        # 计算相对于雷达的位置
        x_rel = d_h * math.cos(yaw)
        y_rel = d_h * math.sin(yaw)
        # print("rel: ", x_rel, y_rel)

        # 转换为场地坐标系中的绝对位置
        x_abs = x_rel + radarPosition[0]  # 初始雷达位置是负的
        y_abs = y_rel + radarPosition[1]

        # print("abs:", x_abs, y_abs)

        self.xLocation = x_abs
        self.yLocation = y_abs

    # 实际连上服务器之后不使用这个
    # 这部分只是显示，不写在类里面了
    def transformToImage(self):
        # 将实际坐标转换为图片上的像素坐标
        x_abs, y_abs = self.xLocation, self.yLocation
        x_pixel = x_abs / actual_width * imgSize[0]
        y_pixel = y_abs / actual_height * imgSize[1]
        # print("pixel:", x_pixel, y_pixel)
        return x_pixel, y_pixel

    def getTargetImageCoordinates(self):
        x_abs, y_abs = calculate2DPosition(self.box, self.distance)
        x_pixel, y_pixel = transformToImage(x_abs, y_abs)
        return x_pixel, y_pixel
