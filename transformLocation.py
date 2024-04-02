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


# yaw 偏航角 pitch俯仰角
def calculateYawPitch(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_image_center = frameSize[0] / 2
    y_image_center = frameSize[1] / 2

    # 计算偏航角
    yaw = ((x_center - x_image_center) / frameSize[0]) * fov_horizontal

    # 计算俯仰角
    pitch = ((y_center - y_image_center) / frameSize[1]) * fov_vertical

    return yaw, pitch


def calculate2DPosition(box, distance):
    # pitch暂时没用因为已经有了高度不需要用pitch
    # 但是如果计算高台可能需要
    # 目前看起来在己方高地上 10m以内误差绝对水平（地面）误差在0.6m左右，大致可以接受，因此高台转换优先级不高
    yaw, pitch = calculateYawPitch(box)

    # 将角度转换为弧度
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)

    d_h = max((distance * 100) ** 2 - radarPosition[2] ** 2, 0) ** 0.5  # 暂时无法处理距离小于4m的
    # 同时想到可以用这个距离的角度算车是不是在高地上

    # 计算相对于雷达的位置
    x_rel = d_h * math.cos(yaw)
    y_rel = d_h * math.sin(yaw)
    print("rel: ", x_rel, y_rel)

    # 转换为场地坐标系中的绝对位置
    x_abs = x_rel + radarPosition[0]  # 初始雷达位置是负的
    y_abs = y_rel + radarPosition[1]

    print("abs:", x_abs, y_abs)

    return x_abs, y_abs


def transformToImage(x_abs, y_abs):
    # 将实际坐标转换为图片上的像素坐标
    x_pixel = x_abs / actual_width * imgSize[0]
    y_pixel = y_abs / actual_height * imgSize[1]
    # print("pixel:", x_pixel, y_pixel)
    return x_pixel, y_pixel


def getTargetImageCoordinates(box, distance):
    x_abs, y_abs = calculate2DPosition(box, distance)
    x_pixel, y_pixel = transformToImage(x_abs, y_abs)
    return x_pixel, y_pixel
