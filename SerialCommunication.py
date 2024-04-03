import serial
import time


# 串口配置类
class SerialCommunication:
    def __init__(self, port, baudrate=115200, bytesize=8, stopbits=1, parity='N', timeout=None):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.bytesize = bytesize
        self.ser.stopbits = stopbits
        self.ser.parity = parity
        self.ser.timeout = timeout
        self.ser.xonxoff = False  # 禁用软件流控
        self.ser.rtscts = False  # 禁用硬件流控
        self.ser.dsrdtr = False  # 禁用硬件流控

    def open_serial(self):
        if not self.ser.is_open:
            self.ser.open()

    def close_serial(self):
        if self.ser.is_open:
            self.ser.close()

    def send_data(self, data):
        self.ser.write(data)

    def receive_data(self):
        while True:
            if self.ser.in_waiting > 0:
                data = self.ser.read_all()
                return data


def send_packet(serial_port, packet_builder, rate_hz=10):
    # 计算发送间隔时间（单位是秒）
    interval = 1.0 / rate_hz

    # 打开串口
    if not serial_port.is_open:
        serial_port.open()

    # 主发送循环
    try:
        while True:
            start_time = time.time()

            packet = packet_builder.build_packet()

            serial_port.write(packet)

            end_time = time.time()

            # 计算实际发送所需时间，并等待剩余的时间间隔
            elapsed_time = end_time - start_time
            time_to_wait = interval - elapsed_time

            if time_to_wait < 0:
                print("Warning: Sending took longer than the desired interval.")
            else:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("Stopping packet sending.")

    finally:
        serial_port.close()


def receive_packet(serial_port, packet_parser, rate_hz=5):
    interval = 1.0 / rate_hz

    # 打开串口
    if not serial_port.is_open:
        serial_port.open()

    try:
        while True:
            start_time = time.time()

            # 检查是否有等待的数据
            if serial_port.in_waiting:
                packet = serial_port.read_all()

                # 解析收到的数据包
                try:
                    seq, cmd_id, data = packet_parser.parse_packet(packet)
                    parsed_data = packet_parser.analyse_data(cmd_id, data)
                    print(f"Received packet: {parsed_data}")
                except ValueError as e:
                    print(f"Error parsing packet: {e}")

            elapsed_time = time.time() - start_time
            time_to_wait = interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("Stopping packet reception.")

    finally:
        # 关闭串口
        serial_port.close()
