import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(PWMA,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)
GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)

L_Motor = GPIO.PWM(PWMA,500)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB,500)
R_Motor.start(0)

def motor_go(speed):
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def motor_left(speed):
    GPIO.output(AIN1,1)
    GPIO.output(AIN2,0)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def motor_right(speed):
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,1)
    GPIO.output(BIN2,0)
    R_Motor.ChangeDutyCycle(speed)

def motor_stop():
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(0)


def main():

    """
    # old settings
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
    """
    
    # new settings
    height = 480
    width = 640
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"format": 'XRGB8888',
                                                           "size": (width, height)}))
    camera.start()
    
    while( True ):
        #success, image = camera.read() #image is in the form of 3d matrix
        image = camera.capture_array()
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])   # lower HSV for blue
        upper_blue = np.array([140, 255, 255])  # upper HSV for blue
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        masked = cv2.bitwise_and(image, image, mask=mask)
        #result = image - masked
        
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None
        for c in contours:
        	area = cv2.contourArea(c) # returns number of pixels within the area
        	thresholdpixels = 2000
        	
        	if (area > max_area and area > thresholdpixels):
        		max_area = area
        		max_contour = c
        
        x, y, w, h = cv2.boundingRect(max_contour)
        centroid_x = int(x+w/2)
        centroid_y = int(y+h/2)
        print(centroid_x, "," , centroid_y)
        cv2.circle(image, (centroid_x,centroid_y), 10, (0, 0, 255), -1)

        cv2.imshow("original video", image)
        cv2.imshow('output video',masked)   
        
        print("max area is: ", max_area)
        if(centroid_x > 0 and centroid_x < 214):
        	print("turn left")
        	motor_left(50)
        elif(centroid_x > 214 and centroid_x < 426):
        	print("go straight")
        	motor_go(50)
        	# count how many number of nonzero pixels in mask
        	count = np.count_nonzero(mask)
        	print("count: ", count)
        	if count > 5000:
        		print("halt!")
        		motor_stop()
        elif(centroid_x > 426 and centroid_x < 640):
        	print("go right")
        	motor_right(50)
        else:
        	print("cannot detect target")
        	motor_stop()

        if cv2.waitKey(1) == ord('q'):
            print("quiting")
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
