import struct
from enum import IntEnum
from CRC import *


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


# 以下解包命令中packet均为解包之后去掉包头的data

# 0x0001
class GameStatus:
    def __init__(self, packet):
        unpacked_data = struct.unpack('>BHBQ', packet)

        self.game_type = (unpacked_data[0] & 0xF0) >> 4
        self.game_progress = unpacked_data[0] & 0x0F

        self.stage_remain_time = unpacked_data[1]
        self.sync_timestamp = unpacked_data[2]


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


# 雷达状态，易伤状态 server to radar
# 0x020C
class RadarInfo:
    def __init__(self, packet):
        self.packet, = struct.unpack('>B', packet)

    # 剩余易伤次数
    def vulnerability_times(self):
        return self.packet & 0b11

    # 0:敌方未被触发双倍易伤
    # 1:敌方正在被触发双倍易伤
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


# 发送机器人坐标包  radar to server
# 0x0305
class MapRobotData:
    def __init__(self, target_robot_id, target_position_x, target_position_y):
        self.cmd_id = 0x0305
        self.target_robot_id = target_robot_id
        self.target_position_x = target_position_x
        self.target_position_y = target_position_y

        self.data = struct.pack('>Hff', self.target_robot_id, self.target_position_x, self.target_position_y)
        # self.packet_data = bytearray(self.data)


class PacketBuilder:
    def __init__(self, cmd_id, seq, data):
        self.sof = 0xA5  # 固定起始位
        self.cmd_id = cmd_id
        self.seq = seq
        self.data = bytearray(data)
        self.data_length = len(bytearray(data))

        self.header = append_crc8_check_sum(bytearray(struct.pack('>BHB', self.sof, self.data_length, self.seq)))
        self.packet_data = self.header + self.data
        self.message = append_crc16_check_sum(self.packet_data)

class PacketParser:
    def __init__(self, packet):
        self.packet = packet

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
        sof, data_length, seq, crc8 = struct.unpack('>BHB', packet[:5])


if __name__ == "__main__":
    mapRobotData = MapRobotData(101, 123.456, 456.789)
    print("mapRobotData.data: ", mapRobotData.data)
    packet = PacketBuilder(mapRobotData.cmd_id, 1, mapRobotData.data)
    print("packet.header: ", packet.header)
    print("header len: ", len(packet.header))
    print("packet data: ", packet.data)
    print("packet data len: ", len(packet.data))
    print("packet message: ", packet.message)
