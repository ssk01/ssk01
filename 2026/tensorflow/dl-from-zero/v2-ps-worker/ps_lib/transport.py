"""3.3 通信层: TCP + pickle 长度分帧

对应 TF 的 Rendezvous / RpcRecvTensor: 跨进程张量传输。
单机多进程用 TCP 即可; 换到真多机时协议不变, 只改地址。
消息: {"type": "...", ...}, 序列化后 4 字节大端长度前缀。
"""
import pickle
import socket
import struct


def send_msg(sock, msg):
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    n = struct.unpack(">I", head)[0]
    data = _recv_exact(sock, n)
    if data is None:
        return None
    return pickle.loads(data)
