import threading
import hikcam  # 假设 hikcam 是用于与摄像头交互的库

class CameraHandler:
    def __init__(self):
        self.framexxx = None
        self.con = threading.Condition()
        self.cap = hikcam.HikCam()
        self.running = False

    def start_camera(self):
        self.cap.start_camera()
        self.cap.set_camera(15, 7000)
        self.con.acquire()

    def stop_camera(self):
        self.running = False
        self.con.acquire()
        self.con.notify_all()
        self.con.release()
        self.cap.close_device()

    def get_img(self):
        self.running = True
        self.start_camera()
        self.con.acquire()

        while self.running:
            self.framexxx = self.cap.get_image(False)
            self.con.notify()
            self.con.wait()

        self.con.notify_all()
        self.con.release()

    def get_frame(self):
        if self.framexxx is not None:
            tempFrame = self.framexxx.copy()
            self.framexxx=None
            return tempFrame
        else:
            self.con.notify_all()
        return None

    def start_thread(self):
        t = threading.Thread(target=self.get_img)
        t.daemon = True
        t.start()