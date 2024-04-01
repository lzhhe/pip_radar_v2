import struct
from enum import IntEnum
import crcmod


# 比赛类型枚举
class GameType(IntEnum):
    RMUL = 1  # 超级对抗赛，剩下的都没用写了防止报错
    TEMP1 = 2
    TEMP2 = 3
    TEMP3 = 4
    TEMP4 = 5


# 比赛阶段枚举
class GameProgress(IntEnum):
    NOT_STARTED = 0  # 未开始
    PREPARATION = 1  # 准备阶段
    SELF_CHECKING_15SECONDS = 2  # 15秒自检
    COUNTDOWN_5SECONDS = 3  # 5秒倒计时
    IN_COMPETITION = 4  # 比赛中
    FINISHED = 5  # 结束


# 0x0001
class GameStatus:
    def __init__(self, packet):
        unpacked_data = struct.unpack('>BHBQ', packet)

        self.game_type = (unpacked_data[0] & 0xF0) >> 4
        self.game_progress = unpacked_data[0] & 0x0F

        self.stage_remain_time = unpacked_data[1]
        self.sync_timestamp = unpacked_data[2]

    # 比赛类型
    def get_game_type_enum(self):
        return GameType(self.game_type)

    # 比赛阶段
    def get_game_progress_enum(self):
        return GameProgress(self.game_progress)

    # 比赛剩余时间
    def get_game_remain_time(self):
        return self.stage_remain_time

    # UNIX 时间戳
    def get_stamp_time(self):
        return self.sync_timestamp


# 1:红方英雄机器人
# 2:红方工程机器人
# 3/4/5:红方步兵机器人（与机器人 ID 3~5 对应）
# 7:红方哨兵机器人

# 101:蓝方英雄机器人
# 102:蓝方工程机器人
# 103/104/105:蓝方步兵机器人（与机器人 ID 3~5 对应）
# 107:蓝方哨兵机器人

class RedRobotID(IntEnum):
    RED_HERO = 1
    RED_ENGINEER = 2
    RED_INFANTRY_3 = 3
    RED_INFANTRY_4 = 4
    RED_INFANTRY_5 = 5
    RED_SENTRY = 7


class BlueRobotID(IntEnum):
    BLUE_HERO = 101
    BLUE_ENGINEER = 102
    BLUE_INFANTRY_3 = 103
    BLUE_INFANTRY_4 = 104
    BLUE_INFANTRY_5 = 105
    BLUE_SENTRY = 107


# 以下解包命令中packet均为解包之后去掉包头的data

# 雷达状态，易伤状态 server to radar
# 0x020C
class RadarInfo:
    def __init__(self, packet):
        self.packet, = struct.unpack('>B', packet)

    def vulnerability_times(self):
        return self.packet & 0b11

    # 0:对方未被触发双倍易伤
    # 1:对方正在被触发双倍易伤
    def vulnerability_status(self):
        return (self.packet >> 2) & 0b1


# 标记进度 server to radar
# 0x020E
class RadarMarkData:
    def __init__(self, packet):
        self.mark_hero_progress, \
            self.mark_engineer_progress, \
            self.mark_infantry_3_progress, \
            self.mark_infantry_4_progress, \
            self.mark_infantry_5_progress, \
            self.mark_sentry_progress = struct.unpack('6B', packet)

    def get_hero_progress(self):
        return self.mark_hero_progress

    def get_engineer_progress(self):
        return self.mark_engineer_progress

    def get_standard_3_progress(self):
        return self.mark_infantry_3_progress

    def get_standard_4_progress(self):
        return self.mark_infantry_4_progress

    def get_standard_5_progress(self):
        return self.mark_infantry_5_progress

    def get_sentry_progress(self):
        return self.mark_sentry_progress


# 发送机器人坐标包  radar to server
# 0x0305
class MapRobotData:
    def __init__(self, target_robot_id, target_position_x, target_position_y):
        self.cmd_id = 0x0305
        self.target_robot_id = target_robot_id
        self.target_position_x = target_position_x
        self.target_position_y = target_position_y

    def pack(self):
        return struct.pack('>Hff', self.target_robot_id, self.target_position_x, self.target_position_y)

    def __repr__(self):
        return f'MapRobotData({self.target_robot_id}, {self.target_position_x}, {self.target_position_y})'


class PacketBuilder:
    def __init__(self, cmd_id, seq, data):
        self.sof = 0xA5  # Start of Frame byte as per specification
        self.cmd_id = cmd_id
        self.seq = seq
        self.data = data
        self.data_length = len(data)

        # Define CRC-8 and CRC-16 functions according to your polynomial
        # This is just a placeholder and needs to be adjusted to your CRC algorithm
        self.crc8_func = crcmod.mkCrcFun(0x107, initCrc=0x00, xorOut=0x00)
        self.crc16_func = crcmod.mkCrcFun(0x11021, initCrc=0x00, xorOut=0x00)

    def calculate_crc8(self, header):
        return self.crc8_func(header)

    def calculate_crc16(self, frame):
        return self.crc16_func(frame)

    def build_packet(self):
        # Construct the frame header
        header = struct.pack('>BHB', self.sof, self.data_length, self.seq)
        crc8 = self.calculate_crc8(header)

        # Include the CRC8 in the header
        header += struct.pack('>B', crc8)

        # Construct the frame tail with CRC16
        frame_without_crc16 = header + self.cmd_id + self.data
        crc16 = self.calculate_crc16(frame_without_crc16)

        # Include the CRC16 in the frame tail
        frame_tail = struct.pack('>H', crc16)

        # Complete frame
        return frame_without_crc16 + frame_tail


class PacketParser:
    def __init__(self):
        self.crc8_func = crcmod.mkCrcFun(0x107, initCrc=0x00, xorOut=0x00)
        self.crc16_func = crcmod.mkCrcFun(0x11021, initCrc=0x00, xorOut=0x00)

        # 创建命令ID到处理函数的映射
        self.cmd_id_map = {
            0x0001: GameStatus,
            0x020C: RadarInfo,
            0x020E: RadarMarkData,
        }

    def parse_packet(self, packet):
        if len(packet) < 7:
            raise ValueError("Incomplete packet: too short for header, CRC8, and CRC16.")

        # 解析包头
        sof, data_length, seq, crc8_received = struct.unpack('>BHB', packet[:5])
        if sof != 0xA5:
            raise ValueError("Invalid start of frame byte.")

        if crc8_received != self.crc8_func(packet[:4]):
            raise ValueError("CRC8 verification failed.")

        data_end_pos = 5 + data_length
        if len(packet) < data_end_pos + 2:
            raise ValueError("Incomplete packet: Data length does not match.")

        crc16_received = struct.unpack('>H', packet[data_end_pos:data_end_pos + 2])[0]
        if crc16_received != self.crc16_func(packet[:data_end_pos]):
            raise ValueError("CRC16 verification failed.")

        # 解析cmd id 和数据
        cmd_id = struct.unpack('>H', packet[5:7])[0]
        data = packet[7:data_end_pos]

        return self.analyse_data(cmd_id, data)

    def analyse_data(self, cmd_id, data):
        if cmd_id in self.cmd_id_map:
            return self.cmd_id_map[cmd_id](data)
        else:
            print(f"Unknown command ID: {cmd_id}")
            return None
