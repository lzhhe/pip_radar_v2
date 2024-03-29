import socket
import struct

ip_address = '192.168.1.1'
port = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

values = (
    0,  # t_mark_hero_progress
    1,  # t_mark_engineer_progress
    2,  # t_mark_standard_3_progress
    3,  # t_mark_standard_4_progress
    4,  # t_mark_standard_5_progress
    5,  # t_mark_sentry_progress
)

packed_data = struct.pack('6B', *values)

sock.sendto(packed_data, (ip_address, port))

sock.close()
