import socket


# 子节长度
HEADER = 64

PORT = 5050
DISCONNECTIO_MSG= "disconnect"
# win下
# SERVER = "192.168.1.27"
#mac
SERVER = "127.0.0.1"
ADDR = (SERVER,PORT)
FORMAT = 'utf-8'

client  = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(ADDR)


def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)

    send_length = str(msg_length).encode(FORMAT)

    send_length+=b' ' *(HEADER- len(send_length))

    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

send("hello from socket python")
send("hello from socket-2")
send("hello from socket-3")
send(DISCONNECTIO_MSG)

