import socket
import struct
import pickle
import cv2
import artemis_autonomous_car
import time

server_address = ('192.168.0.114', 20003)
show_info=True

def send_control(control_giro,control_acelerador,address):
    sock.sendto(struct.pack('c',bytes('C','ascii'))+struct.pack('d',round(control_giro,3))+struct.pack('d',round(control_acelerador,3)),address)

if __name__ == "__main__":
    
    cuenta = 0
    received_payload=b''
    auto_utils=artemis_autonomous_car.artemis_autonomous_car([2,3,2,3,2,2,2,2,0],0)#[2,3,1,3,2,2,0][2,3,2,3,2,2,2,2,0]
    #Generación de socket
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
            control_giro,control_acelerador,trayectory_not_found=auto_utils.proceso_fotograma(img,show_info)
            send_control(control_giro,control_acelerador,address)
        if data_type == b'D':
            pass
        if data_type == b'L':
            #print("Información LIDAR recibida!")
            auto_utils.proceso_lidar(data,False)
        if data_type == b'B':
            auto_utils.set_battery_level(data)
    #except Exception:
    #    pass
    #finally:
        # Clean up the connection
    #    pass#connection.close()
