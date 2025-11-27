import time
import csv
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from motor_controller import MotorController

# ===================== GPIO CHECK =====================
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


# ===================== SERVO =====================
factory = LGPIOFactory()
servo = Servo(
    13,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory
)

def angle_to_servo_value(angle):
    return (angle - 90) / 90.0


# ===================== ULTRASONIC =====================
TRIG = 5
ECHO = 6
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def measure_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    return round((pulse_end - pulse_start) * 17150, 2)


# ===================== SERVO SCAN =====================
def scan_with_servo():
    scan_angles = [150, 90, 30]
    distances = {}

    for ang in scan_angles:
        servo.value = angle_to_servo_value(ang)
        time.sleep(0.15)
        dist = measure_distance()
        distances[ang] = dist
        print(f"SCAN {ang}° → {dist} cm")

    return distances


def choose_best_direction(motor):
    distances = scan_with_servo()
    best_angle = max(distances, key=distances.get)

    print("BEST ANGLE:", best_angle)

    if best_angle == 90:
        motor.move_forward(40)
        time.sleep(0.4)
    elif best_angle == 150:
        motor.turn_in_place(-1, 50, 0.45)
    elif best_angle == 30:
        motor.turn_in_place(1, 50, 0.45)

    motor.stop()


# ===================== MAIN =====================
def main():
    motor = MotorController()

    width, height = 640, 480
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(
        main={"format": 'XRGB8888', "size": (width, height)}
    ))
    camera.start()

    OBSTACLE_LIMIT = 40
    CLOSE_OBJECT_AREA = 60000
    SPEED = 40

    servo.value = angle_to_servo_value(90)

    while True:
        frame = camera.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        servo.value = angle_to_servo_value(90)
        time.sleep(0.03)
        distance = measure_distance()

        print("FRONT DIST:", distance)

        if distance < OBSTACLE_LIMIT:
            print("OBSTACLE → SERVO SCAN")
            motor.stop()
            choose_best_direction(motor)
            continue

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)

            if area > 2000:
                x, y, w, h = cv2.boundingRect(max_contour)
                cx = int(x + w / 2)

                print("BLUE AREA:", area)

                if area > CLOSE_OBJECT_AREA:
                    print("TARGET REACHED → NEW SCAN")
                    motor.stop()
                    choose_best_direction(motor)
                    continue

                if cx < width * 0.4:
                    motor.set_speed(-SPEED, SPEED)
                elif cx > width * 0.6:
                    motor.set_speed(SPEED, -SPEED)
                else:
                    motor.move_forward(SPEED)
            else:
                motor.stop()
        else:
            motor.turn_in_place(-1, 35, 0.2)

        if cv2.waitKey(1) == ord('q'):
            break

    motor.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
