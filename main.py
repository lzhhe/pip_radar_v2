import os
from multiprocessing import Pipe, Process, set_start_method

import numpy as np

from kmeans import KmeansCalculate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import threading
import cv2
import hikcam

from ultralytics import YOLO

from fps_counter import FPSCounter
from container import Container

# 超参数和配置
use_video = True
pt_file = "huazhongv8.pt"
test_video_file = "test2.mp4"
enemy = "blue"
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


def frameProcess(trackerPipe) -> None:
    print("enter frameProcess")
    if use_video == True:
        cap = cv2.VideoCapture(test_video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            trackerPipe.send(frame)
            # cv2.imshow("frame", frame)
            print("send frame in pipe")
        cap.release()
        cv2.destroyAllWindows()
        return

    else:
        fps_counter = FPSCounter()
        global framexxx
        global con

        con.acquire()
        t_camera = threading.Thread(target=get_img, args=())
        t_camera.daemon = True
        t_camera.start()
        print("相机开始运行")
        while 1:
            con.wait()
            if framexxx is None:
                con.notify()
                continue
            frame = framexxx.copy()
            con.notify_all()
            fps_counter.tick()
            trackerPipe.send(frame)
            print("FPS:", fps_counter.fps)
        t_camera.join()
        cv2.destroyAllWindows()


# 推理进程 反馈所有识别框
def trackerProcess(trackerPipe, updateLabelPipe) -> None:
    print("enter trackerProcess")
    model = YOLO(pt_file)  # 模型初始化
    while True:
        frame = trackerPipe.recv()
        carDict = {}  # 存储car的检测结果：id : [box, label, conf]
        armorDict = {}  # 存储armor的检测结果：id : [box, label, conf]
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

                # 这里不使用敌人标签进行过滤是因为container在两个车相交时会将标签给两个车，无法区别了，这部分需要优化
                # 现阶段的办法就是都识别，算力似乎足够而且没吃满
                # 如果使用严格的交并比和分数进行过滤，也无法完全保证准确识别
                # 敌我区分则在最后发送时进行过滤
                # elif 'armor' in label_name and enemy in label_name:
                elif 'armor' in label_name:
                    armorDict[id] = [box, label_name, conf]
            # print("carDict: ", carDict)
            # print("armorDict: ", armorDict)


        except Exception as e:
            print(e)

        tempDict = (carDict, armorDict)  # 只发送一个元素
        updateLabelPipe.send(tempDict)


def updateLabelProcess(updateLabelPipe, kmeansContainerPipe) -> None:
    print("enter updateLabelProcess")
    containerDict = {}
    while 1:
        tempContainerDict = {}
        enemyDict = {}
        carDict, armorDict = updateLabelPipe.recv()

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

        # 需要改成敌方的容器
        containerDict.clear()
        containerDict.update(tempContainerDict)

        for id in containerDict:
            if enemy in tempContainerDict[id].label:
                enemyDict[id] = tempContainerDict[id]
        kmeansContainerPipe.send(enemyDict)  # 这后面都只有敌方信息
        # print(enemyDict)


def kmeansProcess(kmeansContainerPipe, kmeansDepthPipe, distancePipe) -> None:
    print("enter kmeansContainerPipe")
    while True:
        containerDict = kmeansContainerPipe.recv()
        # depth_map = kmeansDepthPipe.recv() # 等待深度转化进程的深度图像，这部分需要根据相机的分辨率来改
        depth_map = np.random.randint(0, 256, (1080, 1440)).astype(np.float32)

        for id in containerDict:
            kmeansCalculate = KmeansCalculate(containerDict[id], depth_map)
            kmeansCalculate.kmeans_classify()

        distancePipe.send(containerDict)  # 这里是已经更新过距离的


def transformProcess(distancePipe, locationPipe) -> None:
    print("enter transformProcess")
    while True:
        containerDict = distancePipe.recv()
        for id in containerDict:
            container = containerDict[id]
            container.calculate2DPosition()
        locationPipe.send(containerDict)  # 最终坐标


def resultProcess(locationPipe) -> None:
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
    while True:
        containerDict = locationPipe.recv()
        for id in containerDict:
            container = containerDict[id]
            label = containerDict[id].label
            print(label)
            xLocation = container.xLocation
            yLocation = container.yLocation

            robotId = robotIdDict.get(label)

            # 等待组包
            # 暂时打印所有的目标和实际坐标
            print("robotId: ", robotId, "xLocation: ", xLocation, "yLocation: ", yLocation)


def main() -> None:
    # __init__

    set_start_method('spawn')

    # 主程序 相机图像获取，以及最后的组包发送
    # 推理进程 反馈所有识别框
    # 匹配进程 接受识别框，反馈敌方坐标容器
    # 点云获取进程 获得点云转化成深度图
    # Kmeans进程 使用kmeans进行推理，接受敌方容器列表和深度图 修改容器当中的distance
    # 主程序 final 组包发送，负责决策

    # 通信管道
    trackerPipe_recv, trackerPipe_send = Pipe(duplex=False)
    updateLabelPipe_recv, updateLabelPipe_send = Pipe(duplex=False)
    kmeansContainerPipe_recv, kmeansContainerPipe_send = Pipe(duplex=False)
    kmeansDepthPipe_recv, kmeansDepthPipe_send = Pipe(duplex=False)
    distancePipe_recv, distancePipe_send = Pipe(duplex=False)
    locationPipe_recv, locationPipe_send = Pipe(duplex=False)

    print("pipes init")

    # 进程列表
    processes = [
        Process(target=frameProcess, args=(trackerPipe_send,)),
        Process(target=trackerProcess, args=(trackerPipe_recv, updateLabelPipe_send)),
        Process(target=updateLabelProcess, args=(updateLabelPipe_recv, kmeansContainerPipe_send)),

        Process(target=kmeansProcess, args=(kmeansContainerPipe_recv, kmeansDepthPipe_recv, distancePipe_send)),
        Process(target=transformProcess, args=(distancePipe_recv, locationPipe_send)),
        Process(target=resultProcess, args=(locationPipe_recv,))
    ]

    print("processes init")

    [p.start() for p in processes]
    [p.join() for p in processes]


if __name__ == "__main__":
    main()
