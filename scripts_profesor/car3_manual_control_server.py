import socket
import struct
import pickle
import cv2
#import artemis_autonomous_car
import time
from pynput import keyboard
import numpy as np
import cmath, math

server_address = ('192.168.0.114', 20003)

control_acelerador=0
control_giro=0

def on_press(key):
    global control_acelerador
    global control_giro
    if key==keyboard.KeyCode.from_char('2'):
        control_acelerador=1
    if key==keyboard.KeyCode.from_char('w'):
        control_acelerador=0.6
    if key==keyboard.KeyCode.from_char('s'):
        control_acelerador=-0.5
    if key==keyboard.KeyCode.from_char('x'):
        control_acelerador=-0.9
    if key==keyboard.KeyCode.from_char('a'):
        control_giro=1
    if key==keyboard.KeyCode.from_char('d'):
        control_giro=-1

def on_release(key):
    global control_acelerador
    global control_giro
    if key==keyboard.KeyCode.from_char('2'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('w'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('s'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('x'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('a'):
        control_giro=0.25
    if key==keyboard.KeyCode.from_char('d'):
        control_giro=0.25

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def send_control(control_giro,control_acelerador,address):
    sock.sendto(struct.pack('c',bytes('C','ascii'))+struct.pack('d',round(control_giro,3))+struct.pack('d',round(control_acelerador,3)),address)

if __name__ == "__main__":
    
    cuenta = 0
    received_payload=b''
    #auto_utils=artemis_autonomous_car.artemis_autonomous_car([0])
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
            cv2.imshow("Coche ARTEMIS",img)
            cv2.waitKey(1)
            send_control(control_giro,control_acelerador,address)
        if data_type == b'D':
            pass
        if data_type == b'L':
            img_lidar = np.zeros((480,640,3), np.uint8)
            i=0
            ranges=data
            for range in ranges:
                Z=cmath.rect(range,math.radians(-i-90))
                i = i+1
                if Z.real != float('inf') and Z.real != float('-inf'):
                    x=int(Z.real*70)+320
                else:
                    x=0
                if Z.imag != float('inf') and Z.imag != float('-inf'):
                    y=int(Z.imag*70)+240
                else:
                    y=0
                cv2.circle(img_lidar,(x,y), radius=2, color=(0,0,255),thickness=2)
                
            cv2.imshow("Mapa LIDAR",img_lidar)
            cv2.waitKey(1)
            pass
    #except Exception:
    #    pass   
    #finally:
        # Clean up the connection
    #    pass#connection.close()
