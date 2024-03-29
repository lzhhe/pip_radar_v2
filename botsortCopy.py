import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import threading

import cv2
import time
import numpy as np

import hikcam

from ultralytics import YOLO

from fps_counter import FPSCounter
# from init_camera import buffer
from container import Container


# 超参数和配置
use_video = False
camera_index = 0  # 相机索引
yaml_file = "yolov8s.yaml"
pt_file = "huazhongv8.pt"
tracker_yaml = "botsort.yaml"
test_img_file = "test.jpg"
output_img_file = 'save.mp4'
test_video_file = "test2.mp4"
output_video_file = 'save.mp4'

enemy = "red"

epochs = 100

video_fps = 24

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


def process_video(process_frame):
    if use_video == True:
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

    containerDict = {}

    def process_frame(frame):
        start_time = time.time()  # 开始计时

        # 初始化存储容器
        carDict = {}  # 存储car的检测结果：id : [box, label, conf]
        armorDict = {}  # 存储armor的检测结果：id : [box, label, conf]

        tempContainerDict = {}

        results = model.track(frame, persist=True, tracker="botsort.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        labels = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        try:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, id, label, conf in zip(boxes, ids, labels, confs):
                label_name = model.names[label]
                # 按标签名称分配到不同的字典
                if label_name == 'car':
                    carDict[id] = [box, label_name, conf]
                elif 'armor' in label_name and enemy in label_name:
                    armorDict[id] = [box, label_name, conf]
            # print("carDict: ", carDict)
            # print("armorDict: ", armorDict)

            for id, data in carDict.items():
                box, label_name, conf = data  # 解包carDict中的数据

                if id in containerDict:
                    # print("id已经出现")
                    # 如果id已经存在于containers中，更新box位置
                    tempContainerDict[id] = containerDict.get(id)
                    tempContainerDict[id].box = box  # 更新box
                    tempContainerDict[id].updateLabel(armorDict)
                else:
                    # print("新的id")
                    # 如果id不存在，创建一个新的Container实例并添加到containers字典中
                    tempContainerDict[id] = Container(id, box)
                    tempContainerDict[id].updateLabel(armorDict)
                    # tempContainerDict[id].print_info()


        except Exception as e:
            print(e)

        containerDict.clear()
        containerDict.update(tempContainerDict)
        for container in containerDict.values():
            container.print_info()
            box = container.box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{container.label} Id {container.id} {container.score:.2f} {container.distance:.2f} M"
            cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        end_time = time.time()  # 结束计时
        processing_time = end_time - start_time  # 计算处理时间
        print("处理一帧所需时间: {:.2f} 秒".format(processing_time))  # 打印处理时间


        return frame

    process_video(process_frame)


# 下面是使用netron导出模型结构
# netron.start("YOLOv8/runs/detect/train1/weights/best.onnx")

if __name__ == "__main__":
    print("test tracker")
    tracker()
