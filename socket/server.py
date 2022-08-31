import socket 
import threading 
import time
from xmlrpc.client import Server 
# 子节长度
HEADER = 64

PORT = 5050
# mac-ifconfig本地ip地址
# SERVER = "192.168.3.34"

# 自动获取ip地址
SERVER =socket.gethostbyname(socket.gethostname())


ADDR =(SERVER,PORT)

DISCONNECTIO_MSG= "disconnect"

# 创建af网络，网络协议
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# 套接字链接到网路
server.bind(ADDR)

def handle_client(conn,addr):
    # 并行运行
    print("[new connection]{addr} connected.")
    
    connected = True
    while connected:
        # 接受字节数目
        msg_length = conn.recv(HEADER).decode("utf-8")
        if msg_length:
            msg_length = int(msg_length)

            msg = conn.recv(msg_length).decode("utf-8")
            if msg ==DISCONNECTIO_MSG:
                connected = False
            print(f"[{addr}] {msg}")


    conn.close()


def start():
    # 监听网路
    server.listen()
    print("[listening] server is listenting on {SERVER}")
    while True:
        conn ,addr= server.accept()
        thread = threading.Thread(target=handle_client,args=(conn,addr))
        thread.start()
        print(f"[acctive connection]{threading.activeCount() -1 }")


print("[server]server is staring ...")
start()

