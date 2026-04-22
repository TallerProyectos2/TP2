import socket
import struct
import pickle
import cv2
import artemis_autonomous_car
import time

from pynput import keyboard
auto_utils=0
real_time_control=2
stop=0
def on_press(key):
    global real_time_control
    global auto_utils
    if key==keyboard.KeyCode.from_char('1'):
        real_time_control=1
    if key==keyboard.KeyCode.from_char('2'):
        real_time_control=2
    if key==keyboard.KeyCode.from_char('3'):
        real_time_control=3
    if key==keyboard.KeyCode.from_char('4'):
        real_time_control=4
    if key==keyboard.KeyCode.from_char('6'):
        real_time_control=5
    if key==keyboard.KeyCode.from_char('z'):
        auto_utils.set_stop(1)
    if key==keyboard.KeyCode.from_char('a'):
        auto_utils.set_stop(0)
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

server_address = ('192.168.0.114', 20003)

def send_control(control_giro,control_acelerador,address):
    sock.sendto(struct.pack('c',bytes('C','ascii'))+struct.pack('d',round(control_giro,3))+struct.pack('d',round(control_acelerador,3)),address)

if __name__ == "__main__":
    
    cuenta = 0
    received_payload=b''
    auto_utils=artemis_autonomous_car.artemis_autonomous_car([1,2,3],0)#[1,1,1,1,0][2,2,2,1,2,1,2,1,1,1,0]
    #Generacion de socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.bind(server_address)
    #try:
        # Receive the data in small chunks and retransmit it
    while True:
        #print("Esperando dato")
        data, address = sock.recvfrom(99999)
        #print("Dato recibido")
        data_type = struct.unpack('c',bytes([data[0]]))[0]
        #print("Tipo de dato: ",data_type)
        received_payload=bytes(data[1:])
        data = pickle.loads(received_payload,encoding='latin1')
        if data_type == b'I':
            img=cv2.imdecode(data,1)
            control_giro,control_acelerador,trayectory_not_found=auto_utils.proceso_fotograma(img,True,real_time_control)
            send_control(control_giro,control_acelerador,address)
        if data_type == b'D':
            pass
        if data_type == b'L':
            #print("Informacion LIDAR recibida!")
            auto_utils.proceso_lidar(data,False)
        if data_type == b'B':
            auto_utils.set_battery_level(data)
    #except Exception:
    #    pass
    #finally:
        # Clean up the connection
    #    pass#connection.close()
