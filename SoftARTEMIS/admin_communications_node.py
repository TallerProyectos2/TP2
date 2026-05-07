import ConfigParser
import io
import signal
import time
import socket
import rospy
import paho.mqtt.client as mqtt_client
import pickle
import struct
import cv2
import bmi160
from threading import Lock
from multiprocessing import Process,Queue
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from media_pkg.msg import cameraMSG
from servo_pkg.srv import SetLedCtrlSrv
from servo_pkg.srv import ServoCalSrv
from std_srvs.srv import SetBool
from i2c_pkg.srv import BatteryLevelSrv
from ctrl_pkg.msg import ServoCtrlMsg
from net_artemis import Net_artemis

# Read config file
vehicle_config = ConfigParser.RawConfigParser(allow_no_value=True)
with open("/home/deepracer/SoftARTEMIS/vehicle.conf") as config_file:
	config_data = config_file.read()
vehicle_config.readfp(io.BytesIO(config_data))
vehicle_ID = vehicle_config.get("mqtt","vehicle_ID")
mqtt_server_ip = vehicle_config.get("mqtt","mqtt_server_ip")
mqtt_server_port = vehicle_config.getint("mqtt","mqtt_server_port")
cloud_server_ip = vehicle_config.get("cloud_autonomous_driving","cloud_server_ip")
cloud_server_port = vehicle_config.get("cloud_autonomous_driving","cloud_server_port")
max_calibration = vehicle_config.getint("steering_calibration","max")
mid_calibration = vehicle_config.getint("steering_calibration","mid")
min_calibration = vehicle_config.getint("steering_calibration","min")
stream_server_address = (cloud_server_ip,cloud_server_port)

# Global variables
i = 0
type_conection = "Unknown"
sending_data = 0
lidar_subscription=None
video_subscription=None
sock=None
periodic_process_lanzado=0
MQTT_periodic_process=0
q_MQTT_periodic_sender=Queue()
status = 1

lock=Lock()

def signal_handler(_signo,_stack_frame):
	led_state(red=0,green=0,blue=0)
	sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def MQTT_periodic_sender():
	global type_conection
	print("Proceso lanzado")
	client_MQTT = mqtt_client.Client()
	client_MQTT.connect(mqtt_server_ip,mqtt_server_port,5)
	while 1==1:
		print("ENVIO PERIODICO")
		acc=bmi160.read_accel()
		client_MQTT.publish(str(vehicle_ID)+'/IMU/x',acc[0])
		client_MQTT.publish(str(vehicle_ID)+'/IMU/y',acc[1])
		client_MQTT.publish(str(vehicle_ID)+'/IMU/z',acc[2])
		if type_conection == "Wifi":
			client_MQTT.publish(str(vehicle_ID)+'/WIFI_dBm',Net_artemis.get_dbm_signal_actual())
		client_MQTT.publish(str(vehicle_ID)+'/battery_level',float(str(battery_level()).split()[1]))
		time.sleep(0.5)
		if not q_MQTT_periodic_sender.empty():
			if(q_MQTT_periodic_sender.get()==0):
				client_MQTT.disconnect()
				break


def send_data(tipo,serialized_data):
	with lock:
		sock.sendall(struct.pack('c',tipo)+struct.pack('>I',len(serialized_data))+serialized_data)

def laser_data_stream(msg):
	serialized_data = pickle.dumps(msg.ranges[:])
	send_data('L',serialized_data)

def camera_data_stream(msg):
	global i
	if i == 0:
		img = bridge.imgmsg_to_cv2(msg.images[0],"bgr8")
		
		endoded, img=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),30])
		
		serialized_data = pickle.dumps(img)
		time2 = time.time()
		send_data('I',serialized_data)
		print(time.time()-time2)
		i=0
	else:
		i += 1

def on_connect(client,userdata,flags,rc):
	global MQTT_periodic_process
	global vehicle_ID
	global type_conection
	if rc==0:
		print("Conectado")
		led_state(red=10000000,green=10000000,blue=0)
		client.subscribe(str(vehicle_ID)+'/command')
		client.publish(str(vehicle_ID),"Vehicle "+str(vehicle_ID)+" connected")
		(type_conection,public_IP) = Net_artemis.get_public_IP()
		client.publish(str(vehicle_ID)+'/type_of_conection',type_conection)
		client.publish(str(vehicle_ID)+'/public_ip',public_IP)
		client.publish(str(vehicle_ID)+'/steering_calibration/max',max_calibration)
		client.publish(str(vehicle_ID)+'/steering_calibration/mid',mid_calibration)
		client.publish(str(vehicle_ID)+'/steering_calibration/min',min_calibration)
		client.publish(str(vehicle_ID)+'/mqtt_server_ip',mqtt_server_ip)
		client.publish(str(vehicle_ID)+'/mqtt_server_port',mqtt_server_port)
		client.publish(str(vehicle_ID)+'/cloud_server_ip',cloud_server_ip)
		client.publish(str(vehicle_ID)+'/cloud_server_port',cloud_server_port)
	else:
		print("No conectado")

def on_message(client,userdata,msg):
	global vehicle_config
	global MQTT_periodic_process
	global sending_data
	global periodic_process_lanzado
	global lidar_subscription
	global video_subscription
	global sock
	global status
	
	if msg.payload == "AM-Local":
		print("Recibido mensaje AM-Local")
		enable_cloud_control(data=False)
		enable_local_autonomous_control(data=True)
		led_state(red=0,green=10000000,blue=0)
		status = 2
	
	elif msg.payload == "AM-Cloud":
		print("Recibido mensaje AM-Cloud")
		enable_local_autonomous_control(data=False)
		enable_cloud_control(data=True)
		led_state(red=0,green=0,blue=10000000)
		status = 3

	elif msg.payload == "AM-Off":
		print("Recibido mensaje AM-Off")
		enable_cloud_control(data=False)
		enable_local_autonomous_control(data=False)
		rospy.sleep(0.1)
		pub_manual_drive.publish(angle=0.0,throttle=0.0)
		led_state(red=10000000,green=10000000,blue=0)
		status = 1
	
	elif msg.payload == "Get-Data":
		print("Recibido mensaje Get-Data")
		bmi160.enable_accel()
		client.publish('signal',"0")
		if (periodic_process_lanzado == 0):
			MQTT_periodic_process = Process(target=MQTT_periodic_sender,name="MQTT_periodic_sender")
			MQTT_periodic_process.start()
			periodic_process_lanzado = 1
	
	elif msg.payload == "Stop-Data":
		print("Recibido mensaje Stop-Data")
		if (periodic_process_lanzado == 1):
			q_MQTT_periodic_sender.put(0)
			periodic_process_lanzado = 0

	elif msg.payload[0:9] == "Calibrate":
		try:
			print("Recibido mensaje Calibrate")
			calibration_data=msg.payload.split()[1:]
			calibration_data[0]=int(calibration_data[0])
			calibration_data[1]=int(calibration_data[1])
			calibration_data[2]=int(calibration_data[2])
			calibration(max=calibration_data[0],mid=calibration_data[1],min=calibration_data[2],polarity=1)
			time.sleep(0.1)
			pub_manual_drive.publish(angle=0.0,throttle=0.0)
			led_state(red=0,green=10000000,blue=10000000)
			vehicle_config.set('steering_calibration','max',calibration_data[0])
			vehicle_config.set('steering_calibration','mid',calibration_data[1])
			vehicle_config.set('steering_calibration','min',calibration_data[2])
			with open("/home/deepracer/SoftARTEMIS/vehicle.conf",'w') as config_file:
				vehicle_config.write(config_file)
			client.publish(str(vehicle_ID)+'/steering_calibration/max',calibration_data[0])
			client.publish(str(vehicle_ID)+'/steering_calibration/mid',calibration_data[1])
			client.publish(str(vehicle_ID)+'/steering_calibration/min',calibration_data[2])
			time.sleep(0.5)
		except:
			led_state(red=10000000,green=0,blue=0)
			time.sleep(0.2)
			led_state(red=0,green=0,blue=0)
			time.sleep(0.2)
			led_state(red=10000000,green=0,blue=0)
			time.sleep(0.2)
			led_state(red=0,green=0,blue=0)
			time.sleep(0.5)
		if status == 1:
			led_state(red=10000000,green=10000000,blue=0)
		if status == 2:
			led_state(red=0,green=10000000,blue=0)
		if status == 3:
			led_state(red=0,green=0,blue=10000000)
	elif msg.payload == "get-data" and sending_data == 0:
		#Inicializacion socket
		
		sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect(stream_server_address)
		
		sending_data = 1
		lidar_subscription = rospy.Subscriber('scan',LaserScan,laser_data_stream,queue_size=1)
		video_subscription = rospy.Subscriber('video_mjpeg',cameraMSG,camera_data_stream,queue_size=1,buff_size=2**25)
	
	elif msg.payload == "no-data" and sending_data == 1:
		sending_data = 0
		video_subscription.unregister()
		lidar_subscription.unregister()
		time.sleep(1)
		sock.close()
	else:
		led_state(red=10000000,green=0,blue=0)
		time.sleep(0.2)
		led_state(red=0,green=0,blue=0)
		time.sleep(0.2)
		led_state(red=10000000,green=0,blue=0)
		time.sleep(0.2)
		led_state(red=0,green=0,blue=0)
		time.sleep(0.5)
		if status == 1:
			led_state(red=10000000,green=10000000,blue=0)
		if status == 2:
			led_state(red=0,green=10000000,blue=0)
		if status == 3:
			led_state(red=0,green=0,blue=10000000)

def on_disconnect(client,userdata,rc):
	global periodic_process_lanzado
	global MQTT_periodic_process
	if (periodic_process_lanzado == 1):
			q_MQTT_periodic_sender.put(0)
			periodic_process_lanzado = 0
	led_state(red=10000000,green=0,blue=0)
	enable_cloud_control(data=False)
	enable_local_autonomous_control(data=False)
	rospy.sleep(0.1)
	pub_manual_drive.publish(angle=0.0,throttle=0.0)
	print("Desconectado")

if __name__=='__main__':
	
	print("\n\t\tNODO COMUNICACIONES CON ADMINISTRADOR\n\n");
	#Inicializacion
	rospy.init_node('admin_communications_node')
	bridge = CvBridge()
	
	#Servicios
	led_state = rospy.ServiceProxy('set_led_state',SetLedCtrlSrv)
	calibration = rospy.ServiceProxy('servo_cal',ServoCalSrv)
	battery_level = rospy.ServiceProxy('battery_level', BatteryLevelSrv)
	enable_local_autonomous_control = rospy.ServiceProxy('enable_local_autonomous_control',SetBool)
	enable_cloud_control = rospy.ServiceProxy('enable_cloud_control',SetBool)

	#Publicaciones
	pub_manual_drive=rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=10)
	
	led_state(red=10000000,green=10000000,blue=10000000)
	time.sleep(0.8)
	#Calibracion inicial
	print(max_calibration)
	print(mid_calibration)
	print(min_calibration)
	calibration(max=max_calibration,mid=mid_calibration,min=min_calibration,polarity=1)
	time.sleep(0.3)
	pub_manual_drive.publish(angle=1.0,throttle=0.0)
	time.sleep(0.3)
	pub_manual_drive.publish(angle=-1.0,throttle=0.0)
	time.sleep(0.3)
	pub_manual_drive.publish(angle=0.0,throttle=0.0)
	time.sleep(0.3)
	#Inicializacion MQTT
	
	conectado=0	
	
	client_MQTT = mqtt_client.Client()
	client_MQTT.on_connect = on_connect
	client_MQTT.on_message = on_message
	client_MQTT.on_disconnect = on_disconnect
	while(conectado==0):
		try:
			client_MQTT.connect(mqtt_server_ip,mqtt_server_port,5)
			conectado=1
		except:
			led_state(red=10000000,green=0,blue=0)
			print("No conectado")
			conectado=0
			time.sleep(5)
	client_MQTT.loop_start()
	rospy.spin()
