# 测量帧数 每100真计算一次
import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def tick(self):
        """在每帧处理结束时调用此方法来更新帧计数器"""
        self.frame_count += 1
        if self.frame_count == 100:
            end_time = time.time()
            self.fps = round(self.frame_count / (end_time - self.start_time), 2)
            self.start_time = time.time()
            self.frame_count = 0
            print("FPS:", self.fps)

    def get_fps(self):
        """获取当前FPS值"""
        return self.fps
