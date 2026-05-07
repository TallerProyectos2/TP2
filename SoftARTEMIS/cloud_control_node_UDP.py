import ConfigParser
import io
import signal
import sys
import socket
import json
import os
import rospy
import paho.mqtt.client as mqtt_client
import pickle
import struct
import cv2
import threading
import time
from threading import Timer
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from media_pkg.msg import cameraMSG
from servo_pkg.srv import SetLedCtrlSrv
from std_srvs.srv import SetBool
from ctrl_pkg.msg import ServoCtrlMsg
from i2c_pkg.srv import BatteryLevelSrv
try:
	import bmi160
	BMI160_IMPORT_ERROR = None
except BaseException as exc:
	bmi160 = None
	BMI160_IMPORT_ERROR = str(exc)


# Read config file
vehicle_config = ConfigParser.RawConfigParser(allow_no_value=True)
with open("/home/deepracer/SoftARTEMIS/vehicle.conf") as config_file:
	config_data = config_file.read()
vehicle_config.readfp(io.BytesIO(config_data))
cloud_server_ip = vehicle_config.get("cloud_autonomous_driving","cloud_server_ip")
cloud_server_port = int(vehicle_config.get("cloud_autonomous_driving","cloud_server_port"))
cloud_server_address = (cloud_server_ip,cloud_server_port)

# Global variables
sending_data = 0
lidar_subscription = None
video_subscription = None
encendido = False
sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
i = 0
time2=0
timer_battery_level=0
timer_imu_level=0
imu_seq=0
lock=threading.Lock()
IMU_SEND_INTERVAL_SEC = float(os.environ.get('TP2_IMU_SEND_INTERVAL_SEC','0.1'))

def signal_handler(_signo,_stack_frame):
	
	sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def frame_not_received():
	pub_manual_drive.publish(angle=0,throttle=0)

def send_battery_level():
	global timer_battery_level
	serialized_data = pickle.dumps(float(str(battery_level()).split()[1]))
	send_data('B',serialized_data)
	timer_battery_level=Timer(2,send_battery_level)
	timer_battery_level.start()

def init_imu():
	if bmi160 is None:
		return False
	try:
		bmi160.enable_accel()
		bmi160.enable_gyro()
		return True
	except Exception:
		return False

def read_imu_payload():
	global imu_seq
	imu_seq += 1
	payload = {
		'schema': 'tp2.car.telemetry.v1',
		'seq': imu_seq,
		'ts': time.time(),
		'source': 'SoftARTEMIS',
		'imu': {
			'sensor': 'BMI160',
			'status': 'ok',
		},
	}
	if bmi160 is None:
		payload['imu']['status'] = 'unavailable'
		payload['imu']['error'] = BMI160_IMPORT_ERROR or 'bmi160 import failed'
		return payload
	try:
		acc = bmi160.read_accel()
		bmi160.read_gyro()
		payload['imu']['accel_mps2'] = {
			'x': float(acc[0]),
			'y': float(acc[1]),
			'z': float(acc[2]),
		}
		payload['imu']['gyro_dps'] = {
			'x': float(bmi160.gyro_x),
			'y': float(bmi160.gyro_y),
			'z': float(bmi160.gyro_z),
		}
	except Exception as exc:
		payload['imu']['status'] = 'error'
		payload['imu']['error'] = str(exc)
	return payload

def send_imu_level():
	global timer_imu_level
	if encendido == False:
		return
	payload = read_imu_payload()
	serialized_data = json.dumps(payload,separators=(',',':')).encode('utf-8')
	send_data('D',serialized_data)
	timer_imu_level=Timer(IMU_SEND_INTERVAL_SEC,send_imu_level)
	timer_imu_level.start()

def handle_enable_cloud_control(order):
	global encendido
	global video_subscription
	global lidar_subscription
	global timer_battery_level
	global timer_imu_level
	global sock
	
	if order.data == False and encendido == True:
		print("Cloud control off")
		encendido = False
		video_subscription.unregister()
		lidar_subscription.unregister()
		timer_battery_level.cancel()
		if timer_imu_level != 0:
			timer_imu_level.cancel()
		return [True,'OK']
	
	if order.data == True and encendido == False:
		print("Cloud control on")
		encendido = True
		init_imu()
		video_subscription = rospy.Subscriber('video_mjpeg',cameraMSG,camera_data_stream,queue_size=1,buff_size=2**25)
		lidar_subscription = rospy.Subscriber('scan',LaserScan,laser_data_stream,queue_size=1)
		timer_battery_level=Timer(2,send_battery_level)
		timer_battery_level.start()
		timer_imu_level=Timer(IMU_SEND_INTERVAL_SEC,send_imu_level)
		timer_imu_level.start()
		return [True,'OK']
	
	else:
		return [False,'Already done']

def send_data(tipo,serialized_data):
	with lock:
		sock.sendto(struct.pack('c',tipo)+serialized_data,cloud_server_address)

def send_data_img(tipo,serialized_data):
	print
	UDPMAXBYTES=65000
	with lock:
			sock.sendto(struct.pack('c',tipo)+struct.pack('B',0)+serialized_data[0:UDPMAXBYTES],cloud_server_address)
			sock.sendto(struct.pack('c',tipo)+struct.pack('B',1)+serialized_data[UDPMAXBYTES:(UDPMAXBYTES*2)],cloud_server_address)
			#sock.sendto(struct.pack('c',tipo)+struct.pack('B',2)+serialized_data[(UDPMAXBYTES*2):(UDPMAXBYTES*3)],cloud_server_address)
			#sock.sendto(struct.pack('c',tipo)+struct.pack('B',3)+serialized_data[(UDPMAXBYTES*3):(UDPMAXBYTES*4)],cloud_server_address)

def laser_data_stream(msg):
	serialized_data = pickle.dumps(msg.ranges[:])
	send_data('L',serialized_data)

def camera_data_stream(msg):
	global i
	global time2
	if i == 0:
		
		#time1 = time.time()
		imgL = bridge.imgmsg_to_cv2(msg.images[0],"bgr8")
		#imgR = bridge.imgmsg_to_cv2(msg.images[1],"bgr8")
		imgL=cv2.imencode('.jpg',imgL,[int(cv2.IMWRITE_JPEG_QUALITY),12])[1]
		#imgR=cv2.imencode('.jpg',imgR,[int(cv2.IMWRITE_JPEG_QUALITY),20])[1]
		serialized_dataL = pickle.dumps(imgL)
		#serialized_dataR = pickle.dumps(imgR)
		send_data('I',serialized_dataL)
		#send_data('D',serialized_dataR)
		#time2=time.time()-time1 +time2
		i=0
		#print(time2/counter)
	else:
		i += 1

if __name__=='__main__':
	
	print("\n\t\tNODO CONTROL POR NUBE\n\n");
	timer_frame_not_received=Timer(0.2,frame_not_received)
	#Inicializacion
	rospy.init_node('cloud_control_node')
	bridge = CvBridge()
	
	s = rospy.Service('enable_cloud_control',SetBool,handle_enable_cloud_control)

	#Servicios
	enable_cloud_control = rospy.ServiceProxy('enable_cloud_control',SetBool)
	battery_level = rospy.ServiceProxy('battery_level', BatteryLevelSrv)
	#Publicaciones
	pub_manual_drive=rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=10)
	
	while(1):
		while(encendido == True):
			data, address =sock.recvfrom(99999)
			frame = struct.unpack('c',bytes(data[0]))[0]
			print(type(data[1:9]))
			print(len(data[1:9]))
			#try:
			if frame == 'C':
				timer_frame_not_received.cancel()
				timer_frame_not_received=Timer(0.2,frame_not_received)
				control_giro=struct.unpack('d',data[1:9])[0]
				control_acelerador=struct.unpack('d',bytes(data[9:17]))[0]
				pub_manual_drive.publish(angle=control_giro,throttle=control_acelerador)
				timer_frame_not_received.start()
				print("Giro: ",control_giro,"\nAcelerador: ",control_acelerador)
			#except Exception:
			#	pass
				
		rospy.sleep(0.1)
				
	rospy.spin()
