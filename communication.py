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

        self.remain_time = unpacked_data[1]
        self.sync_timestamp = unpacked_data[3]


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
# 0x020E
class RadarInfo:
    def __init__(self, packet):
        self.packet, = struct.unpack('>B', packet)
        self.vulnerability_times = self.packet & 0b11  # 剩余易伤次数
        self.vulnerability_status = (self.packet >> 2) & 0b1  # 0:敌方未被触发双倍易伤 1:敌方正在被触发双倍易伤


# 标记进度 server to radar
# 0x020C
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


class PacketBuilder:
    def __init__(self, cmd_id, seq, data):
        self.sof = 0xA5  # 固定起始位
        self.cmd_id = cmd_id
        self.seq = seq
        self.data = bytearray(data)
        self.data_length = len(bytearray(data))

        self.cmd_id_bytes = struct.pack('>H', self.cmd_id)

        self.header = append_crc8_check_sum(bytearray(struct.pack('>BHB', self.sof, self.data_length, self.seq)))
        self.message = append_crc16_check_sum(self.header + self.cmd_id_bytes + self.data)


class PacketParser:
    def __init__(self, packet):
        self.packet = packet

        # 创建命令ID到处理函数的映射
        self.cmd_id_map = {
            0x0001: GameStatus,
            0x020C: RadarInfo,
            0x020E: RadarMarkData,
            # 0x0305: MapRobotData  # 这条实际不会存在，测试解包用
        }

        self.sof = self.packet[0]
        if self.sof != 0xA5:
            raise ValueError(f"Invalid SOF; expected 0xA5, got {self.sof:#x}")

        self.data_length, self.seq = struct.unpack('>HB', self.packet[1:4])

        if not verify_crc8_check_sum(self.packet[:5], 5):
            raise ValueError("CRC8 verification failed.")

        self.cmd_id, = struct.unpack('>H', self.packet[5:7])

        self.data = self.packet[7:-2]

        if not verify_crc16_check_sum(self.packet, len(self.packet)):  # CRC: 长度-2已经在校验文件当中写过了
            raise ValueError("CRC16 verification failed.")

        # 这部分是解包用的，给出对应结构的包
        self.payload = self.cmd_id_map.get(self.cmd_id, None)
        if not self.payload:
            raise ValueError(f"Unknown command ID: {self.cmd_id:#x}")
        else:
            self.payload = self.payload(self.data)  # 把纯bytearray 数据给对应解包



if __name__ == "__main__":
    # 发送测试
    mapRobotData = MapRobotData(101, 123.456789, 456.789123)
    print("mapRobotData.data: ", mapRobotData.data)
    packet = PacketBuilder(mapRobotData.cmd_id, 1, mapRobotData.data)
    print("packet.header: ", packet.header)
    print("header len: ", len(packet.header))
    print("packet data: ", packet.data)
    print("packet data len: ", len(packet.data))
    print("packet message: ", packet.message)

    # 解包测试 具体信息需要等裁判服务器进行发送
    receive_packet = PacketParser(packet.message)
    print(f"receive_packet cmd_id: 0x{receive_packet.cmd_id:04X}")
    print("receive_packet data: ", receive_packet.data)
    print("receive_packet seq: ", receive_packet.seq)
