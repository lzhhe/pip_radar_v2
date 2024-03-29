import keyboard

from botsortTest import *


def main_loop():
    print("操作按键:")
    while True:

        if keyboard.is_pressed('t'):
            train()
            break
        elif keyboard.is_pressed('o'):
            onnx()
            break
        elif keyboard.is_pressed('i'):
            test_img()
            break
        elif keyboard.is_pressed('p'):
            predict()
            break
        elif keyboard.is_pressed('v'):
            test_video()
            break
        elif keyboard.is_pressed('k'):
            tracker()
            break
        elif keyboard.is_pressed('q'):
            break


if __name__ == "__main__":
    main_loop()

# 下面是使用netron导出模型结构
# netron.start("YOLOv8/runs/detect/train1/weights/best.onnx")
