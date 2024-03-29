import os
import threading

import cv2
import time
import numpy as np
import psutil

import hikcam

from ultralytics import YOLO

from fps_counter import FPSCounter
from init_camera import buffer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 超参数和配置
use_video = False
camera_index = 0  # 相机索引
yaml_file = "yolov8s.yaml"
pt_file = "huazhongv8.pt"
test_img_file = "test.jpg"
output_img_file = 'save.mp4'
test_video_file = "test2.mp4"
output_video_file = 'save.mp4'
train_data_file = "report.yaml"
epochs = 100

img_size = 1280
video_fps = 24

f_x = f_y = 1000  # 假设的焦距
c_x = img_size / 2  # 图像宽度的一半
c_y = img_size / 2  # 图像高度的一半

# camera_matrix = np.array([[f, 0, cx],
#                               [0, f, cy],
#                               [0, 0, 1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])

# camera_matrix = np.array([
#     [f_x, 0, c_x],
#     [0, f_y, c_y],
#     [0, 0, 1]
# ])

camera_matrix = np.array([
    [1.28935817e+03, 0.00000000e+00, 6.52933651e+02],
    [0.00000000e+00, 1.28544079e+03, 4.88324670e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# dist_coeffs = np.zeros((5, 1))  # 也还没标定
dist_coeffs = np.array([[-0.09254919, 0.0935931, -0.00351003, -0.00365731, 0.33110822]])

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
    "armor7": np.array([  # 哨兵
        [-0.5, -0.25, 0],  # 左上角
        [0.5, -0.25, 0],  # 右上角
        [0.5, 0.25, 0],  # 右下角
        [-0.5, 0.25, 0]  # 左下角
    ])

}

framexxx = None
con = threading.Condition()


def get_img():
    global con
    global framexxx
    con.acquire()
    cap = hikcam.HikCam()
    cap.start_camera()
    cap.set_camera(15, 7000)
    cap.get_image(False)

    while 1:
        framexxx = cap.get_image(False)
        con.notify()
        con.wait()

    con.notify_all()
    con.release()
    cap.close_device()


# 训练模型，没测试
def train():
    model = YOLO(yaml_file)
    model.train(data=train_data_file, epochs=epochs, imgsz=img_size)


# 导出为 ONNX
def onnx():
    model = YOLO(pt_file)
    model.export(format='onnx')


# 测试图片
def test_img():
    model = YOLO(pt_file)
    img = cv2.imread(test_img_file)
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(output_img_file, ann)


# 性能评估
def predict():
    model = YOLO(pt_file)
    metrics = model.val()
    print(metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.maps)


def process_video(process_frame):
    if use_video == True:  # 迈德威视
        cap = cv2.VideoCapture(test_video_file)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, size)

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            # process_frame在每个函数当中不同
            result_frame = process_frame(frame)
            cv2.imshow("Frame", result_frame)
            out.write(result_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    else:
        fps_counter = FPSCounter()
        global framexxx
        global con

        con.acquire()

        # size = (1440, 1080)
        # out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, size)

        t_camera = threading.Thread(target=get_img, args=())
        t_camera.daemon = True
        t_camera.start()
        print("开始运行")

        while 1:

            con.wait()
            if framexxx is None:
                con.notify()
                continue
            frame = framexxx.copy()
            con.notify_all()

            fps_counter.tick()

            # process_frame在每个函数当中不同
            result_frame = process_frame(frame)
            cv2.imshow("Frame", result_frame)
            # out.write(result_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            print("FPS:", fps_counter.fps)

        t_camera.join()

        cv2.destroyAllWindows()
        return


# 测试视频
def test_video():
    model = YOLO(pt_file)

    def process_frame(frame):
        res = model(frame)
        return res[0].plot()

    process_video(process_frame)


def calculate_distance(object_points, image_points, camera_matrix, dist_coeffs):
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


def tracker():
    model = YOLO(pt_file)
    car_boxes_dict = {}
    car_flag_dict = {}
    car_armor_dict = {}
    car_distance_dict = {}

    def process_frame(frame):

        temp_car_boxes_dict = {}  # 临时存储当前帧的车辆信息
        temp_car_armor_dict = {}  # 临时存储当前帧的车辆信息
        temp_car_distance_dict = {}

        results = model.track(frame, persist=True, tracker="botsort.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        labels = results[0].boxes.cls.cpu().numpy()  # 获取标签
        confs = results[0].boxes.conf.cpu().numpy()

        try:

            ids = results[0].boxes.id.cpu().numpy().astype(int)
            # 先遍历车辆 其实这这样破坏了id的用途 但是可行
            for box, id, label in zip(boxes, ids, labels):
                label_name = model.names[label]
                if label_name == "car":
                    temp_car_boxes_dict[id] = box
                    if id in car_flag_dict:  # 上一帧存在且识别
                        car_flag_dict[id] += 1

                    else:  # 新建
                        car_flag_dict[id] = 1

            for car_id in temp_car_boxes_dict:
                if car_id not in temp_car_armor_dict:  # 给新的ID初始化一个列表
                    temp_car_armor_dict[car_id] = {}

            for box, id, label, conf in zip(boxes, ids, labels, confs):
                label_name = model.names[label]
                if "armor" in label_name:
                    for car_id, car_box in temp_car_boxes_dict.items():
                        if car_flag_dict[car_id] > 50:  # 大于30重新分类
                            # print("需要重新分类")
                            if box[0] >= car_box[0] and box[1] >= car_box[1] and box[2] <= car_box[2] and box[3] <= \
                                    car_box[3]:
                                car_flag_dict[car_id] = 1
                                # 在范围之内不然直接跳过这个车
                                # print("在范围之内")
                                armor_dict = temp_car_armor_dict[car_id]  # 当前遍历这个车的装甲字典
                                # print(armor_dict)
                                if label_name not in armor_dict:
                                    # print("新增")
                                    armor_dict[label_name] = conf
                                    # print("新增完成")
                                else:
                                    # 比较同种装甲板置信度高低
                                    # 只更新置信度更高的装甲板
                                    # 一般不会走到这，但是似乎这里有点问题
                                    if conf > armor_dict[label_name]:  # 同样使用 conf
                                        armor_dict[label_name] = conf
                                        # print("car_id"+car_id+"更换置信度")

                            else:
                                continue
                        else:  # 不需要重新分类
                            if car_id not in car_armor_dict:  # 这一帧新增的一个
                                if box[0] >= car_box[0] and box[1] >= car_box[1] and box[2] <= car_box[2] and box[3] <= \
                                        car_box[3]:  # 这里的条件可以改成IOU的交并比，可以尝试过滤一下重复值，也同时保证出界一点可以被识别
                                    car_flag_dict[car_id] = 1
                                    armor_dict = temp_car_armor_dict[car_id]  # 当前遍历这个车的装甲字典
                                    if label_name not in armor_dict:
                                        armor_dict[label_name] = conf
                                    else:
                                        # 这里可能有两个相同的所以还是写一下重复吧
                                        if conf > armor_dict[label_name]:  # 同样使用 conf
                                            armor_dict[label_name] = conf
                                            # print("car_id"+car_id+"更换置信度")
                            else:
                                temp_car_armor_dict[car_id] = car_armor_dict[car_id]

                            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            # label_text = f"{model.names[label]} Id {id} {conf:.2f}"  # 结合标签和ID
                            # cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for car_id in temp_car_boxes_dict:
                box = temp_car_boxes_dict[car_id]
                armor_dict = temp_car_armor_dict[car_id]

                if armor_dict:
                    # 获取置信度最高的装甲标签和置信度
                    highest_armor_label, highest_conf = max(armor_dict.items(), key=lambda item: item[1])
                    # print(highest_armor_label[:6])
                    if highest_armor_label[:6] in armor_sizes:  # 根据6个字符(就是去掉了后面的红蓝)
                        # print("正确获取大小")1
                        # 定义2D点，装甲板在图像中的位置
                        object_points = armor_sizes[highest_armor_label[:6]].astype(np.float32)  # 确保为浮点数类型

                        box = temp_car_boxes_dict[car_id]
                        image_points = np.array([
                            [box[0], box[1]],  # 左上角
                            [box[2], box[1]],  # 右上角
                            [box[2], box[3]],  # 右下角
                            [box[0], box[3]]  # 左下角
                        ], dtype=np.float32)  # 确保为浮点数类型
                    else:
                        # print("跳过")
                        continue  # 如果没有匹配的类别，跳过这个装甲板

                    # 调用PnP算法计算距离 # 近距离查了30cm 远距离大概50cm左右，未调参状态
                    distance = calculate_distance(object_points, image_points, camera_matrix, dist_coeffs)

                    # 更新距离字典
                    temp_car_distance_dict[car_id] = distance

                    # 显示
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    label_text = f"{highest_armor_label} Id {car_id} {highest_conf:.2f} {distance:.2f} M"  # 结合标签和ID
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                else:

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    label_text = "car"  # 没检测到但是打印出来凑个数
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        except Exception as e:
            print(e)

        car_boxes_dict.clear()  # 清除原有的车辆信息
        car_boxes_dict.update(temp_car_boxes_dict)  # 更新为当前帧的车辆信息
        car_armor_dict.clear()
        car_armor_dict.update(temp_car_armor_dict)
        car_distance_dict.clear()
        car_distance_dict.update(temp_car_distance_dict)

        # for car_id in car_armor_dict:
        #     car_armor_dict[car_id].sort(key=lambda x: x['confidence'], reverse=True)
        print(car_armor_dict)
        # print(len(car_armor_dict))
        print(car_boxes_dict)
        # print(len(car_boxes_dict))
        print(car_flag_dict)
        # print(len(car_flag_dict))
        print(car_distance_dict)

        return frame

    process_video(process_frame)


# 下面是使用netron导出模型结构
# netron.start("YOLOv8/runs/detect/train1/weights/best.onnx")

if __name__ == "__main__":
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])

    print("test tracker")
    tracker()
