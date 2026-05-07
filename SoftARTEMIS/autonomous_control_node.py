import rospy
import imutils
import cv2
import time
import artemis_autonomous_car
from sensor_msgs.msg import LaserScan
from media_pkg.msg import cameraMSG
from ctrl_pkg.msg import ServoCtrlMsg
from servo_pkg.srv import SetLedCtrlSrv
from std_srvs.srv import SetBool
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg
import numpy as np
aac=0
encendido=False
video_subscription=None
lidar_subscription=None
error=0
i=0

def handle_enable_local_autonomous_control(order):
	global error	
	global encendido
	global video_subscription
	global lidar_subscription
	global aac
	
	if order.data == False and encendido == True:
		print("Local autonomous control off")
		encendido = False
		video_subscription.unregister()
		lidar_subscription.unregister()
		return [True,'OK']
	if order.data == True and encendido == False:
		path=[2,2,2,2,2,2,2,2,2,2]
		print("Local autonomous control on")
		aac = artemis_autonomous_car.artemis_autonomous_car(path)
		if error == 0:
			leds(red=0,green=10000000,blue=0)
		else:
			leds(red=10000000,green=0,blue=10000000)
		encendido = True
		video_subscription = rospy.Subscriber('video_mjpeg',cameraMSG,CameraDataReceived,queue_size=1,buff_size=2**25)
		lidar_subscription = rospy.Subscriber('scan',LaserScan,LaserDataReceived,queue_size=1)
		return [True,'OK']
	else:
		return [False,'Already done']
	

def LaserDataReceived(msg):
	global aac
	aac.proceso_lidar(msg.ranges[:],False)
	#print throttleNormalized

def CameraDataReceived(msg):
	global error
	global aceleradorControl
	velocidad=0
	k=1
	
	#Convertimos de msg a imagen opencv
	img = bridge.imgmsg_to_cv2(msg.images[0],"bgr8")
	control_giro,control_acelerador,trayectory_not_found = aac.proceso_fotograma(img,False,0)
	print("giro: "+str(control_giro))
	print("acelerador: "+str(control_acelerador))
	pub_manual_drive.publish(angle=control_giro-0.25,throttle=control_acelerador)
	#Cambio de estado de leds
	if trayectory_not_found==1 and error==0:
		error=1
		leds(red=10000000,green=0,blue=10000000)
	if trayectory_not_found==0 and error==1:
		error=0
		leds(red=0,green=10000000,blue=0)

	
if __name__=='__main__':

	print("\n\t\tNODO CONTROL EN LOCAL\n\n");

	pub_manual_drive=rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=10)
	rospy.init_node('autonomous_control_node')
	
	s = rospy.Service('enable_local_autonomous_control',SetBool,handle_enable_local_autonomous_control)
	
	leds=rospy.ServiceProxy('set_led_state',SetLedCtrlSrv)

	bridge = CvBridge()
	#stereo=cv2.StereoBM_create(numDisparities=16, blockSize=15)

	#for i in range(0,100):
	#	pub.publish(angle=i/100.0,throttle=i/100.0)
	#	rospy.sleep(0.5)
	rospy.spin()
