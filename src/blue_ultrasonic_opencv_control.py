import time
import csv
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from motor_controller import MotorController
from mnist_process import *

import tensorflow as tf
import os
from collections import Counter


# ===================== GPIO CHECK =====================
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    

LOW_BLUE  = np.array([100, 150, 50])
HIGH_BLUE = np.array([140, 255, 255])

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


def recognize_digit_3_frames(camera):
    results = []

    for i in range(3):
        frame = camera.capture_array()
        frame = cv2.flip(frame, -1)

        tensor = preprocess_digit_inside_blue_roi(frame, f"live_{i}")
        if tensor is None:
            print("⚠️ Digit not extracted")
            continue

        digit, conf = infer_digit(tensor)
        print(f"Frame {i}: digit={digit}, conf={conf:.2f}")

        if conf > 0.5:
            results.append(digit)

        time.sleep(0.1)

    if not results:
        print("❌ No valid digit recognized")
        return None

    final_digit = Counter(results).most_common(1)[0][0]
    print("✅ FINAL DIGIT:", final_digit)
    return final_digit


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
    scan_angles = [170, 90, 10]
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
        motor.move_forward(60)
        time.sleep(0.7)
    elif best_angle == 170:
        motor.turn_in_place(-1, 50, 0.45)
    elif best_angle == 10:
        motor.turn_in_place(1, 50, 0.45)

    motor.stop()
    


def main():

    motor = MotorController()

    width, height = 640, 480
    camera = Picamera2()
    
    # ===================== VIDEO RECORD =====================
    VIDEO_DIR = "videos"
    os.makedirs(VIDEO_DIR, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(VIDEO_DIR, f"record_{int(time.time())}.avi")

    video_writer = cv2.VideoWriter(
        video_path,
        fourcc,
        20.0,              # FPS
        (width, height)    # размер кадра
    )


    camera.configure(camera.create_video_configuration(
        main={"format": 'XRGB8888', "size": (width, height)}
    ))
    camera.start()

    OBSTACLE_LIMIT = 60
    CLOSE_OBJECT_AREA = 30000
    SPEED = 50

    servo.value = angle_to_servo_value(90)

    while True:
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            LOW_BLUE,
            HIGH_BLUE)
        
        # --- делаем цветную маску ---
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # --- накладываем маску на оригинал ---
        overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

        # --- ПИШЕМ КАДР В ВИДЕО ---
        video_writer.write(overlay)



        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ===== 1. ЕСЛИ КВАДРАТ НАЙДЕН → РУЛИМ ПО КАМЕРЕ =====
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)

            if area > 1000:
                x, y, w, h = cv2.boundingRect(max_contour)
                cx = int(x + w / 2)

                print("BLUE AREA:", area)

                # ✅✅✅ ВОТ ТУТ НОВАЯ ЛОГИКА ✅✅✅
                if area > CLOSE_OBJECT_AREA:
                    print("📸 TARGET REACHED → DIGIT RECOGNITION")

                    motor.stop()
                    time.sleep(0.001)
                    digit = recognize_digit_3_frames(camera)

                    if digit is not None:
                        if digit % 2 == 0:
                            print("↩ EVEN → TURN RIGHT")
                            motor.turn_in_place(1, 60, 0.7)
                        else:
                            print("↪ ODD → TURN LEFT")
                            motor.turn_in_place(-1, 60, 0.7)
                    else:
                        print("⚠️ DIGIT FAIL → SERVO SCAN")
                        choose_best_direction(motor)

                    continue

                if cx < width * 0.4:
                    motor.set_speed(SPEED, -SPEED)
                elif cx > width * 0.6:
                    motor.set_speed(-SPEED, SPEED)
                else:
                    motor.move_forward(SPEED)
                continue

        # ===== 2. ЕСЛИ КВАДРАТА НЕТ → ЕДЕМ ПО СЕНСОРУ =====
        servo.value = angle_to_servo_value(90)
        time.sleep(0.02)
        distance = measure_distance()
        print("NO TARGET → SENSOR DIST:", distance)

        if distance < OBSTACLE_LIMIT:
            print("OBSTACLE (NO TARGET) → SERVO SCAN")
            motor.stop()
            choose_best_direction(motor)
        else:
            motor.move_forward(SPEED)

        if cv2.waitKey(1) == ord('q'):
            break

    motor.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
