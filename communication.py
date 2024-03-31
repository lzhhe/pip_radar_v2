import struct
from enum import IntEnum

import crcmod


# 1:红方英雄机器人
# 2:红方工程机器人
# 3/4/5:红方步兵机器人（与机器人 ID 3~5 对应）
# 7:红方哨兵机器人

# 101:蓝方英雄机器人
# 102:蓝方工程机器人
# 103/104/105:蓝方步兵机器人（与机器人 ID 3~5 对应）
# 107:蓝方哨兵机器人

# 雷达状态，易伤状态 server to radar
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
class RadarMarkData:
    def __init__(self, packet):
        self.mark_hero_progress, \
            self.mark_engineer_progress, \
            self.mark_standard_3_progress, \
            self.mark_standard_4_progress, \
            self.mark_standard_5_progress, \
            self.mark_sentry_progress = struct.unpack('6B', packet)

    def get_hero_progress(self):
        return self.mark_hero_progress

    def get_engineer_progress(self):
        return self.mark_engineer_progress

    def get_standard_3_progress(self):
        return self.mark_standard_3_progress

    def get_standard_4_progress(self):
        return self.mark_standard_4_progress

    def get_standard_5_progress(self):
        return self.mark_standard_5_progress

    def get_sentry_progress(self):
        return self.mark_sentry_progress

# 发送机器人坐标包  to server
class MapRobotData:
    def __init__(self, target_robot_id, target_position_x, target_position_y):
        self.target_robot_id = target_robot_id
        self.target_position_x = target_position_x
        self.target_position_y = target_position_y

    def pack(self):
        # The format string 'Hff' stands for:
        # H: unsigned short (2 bytes, uint16_t)
        # f: float (4 bytes) for target_position_x
        # f: float (4 bytes) for target_position_y
        # '>' specifies big-endian byte order. If your system uses little-endian, change it to '<'.
        return struct.pack('>Hff', self.target_robot_id, self.target_position_x, self.target_position_y)

    def __repr__(self):
        return f'MapRobotData({self.target_robot_id}, {self.target_position_x}, {self.target_position_y})'


# 比赛类型枚举
class GameType(IntEnum):
    ROBOMASTER_UNIVERSITY_COMBAT = 1
    ROBOMASTER_YOUTH_COMBAT = 2
    ICRA_ROBOMASTER = 3
    ROBOMASTER_UNIVERSITY_ENGINEERING = 4
    ROBOMASTER_YOUTH_ENGINEERING = 5


# 比赛阶段枚举
class GameProgress(IntEnum):
    NOT_STARTED = 0
    PREPARATION = 1
    SETUP = 2
    IN_PROGRESS = 3
    PAUSED = 4
    FINISHED = 5


# 比赛状态枚举
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
        # 比赛剩余时间

    def get_game_progress_enum(self):
        return GameProgress(self.game_progress)

    def get_game_remain_time(self):
        return self.stage_remain_time

    def get_stamp_time(self):
        return self.sync_timestamp


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