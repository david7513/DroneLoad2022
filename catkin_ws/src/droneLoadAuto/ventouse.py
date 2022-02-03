#!/usr/bin/python

import Jetson.GPIO as GPIO
import rospy
import sys	
import mavros
from mavros_msgs.msg import RCIn

rospy.init_node('ventouse')

#GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(13,GPIO.OUT)

def checkAux(data):
	#print data.channels[5]
	if data.channels[5] > 1800:
		GPIO.output(13,GPIO.HIGH)
		#print "drop" 
	else:
		GPIO.output(13,GPIO.LOW)
		#print "low"

rospy.Subscriber("/mavros/rc/in",RCIn,checkAux)

if __name__ == '__main__':
	rospy.spin()
