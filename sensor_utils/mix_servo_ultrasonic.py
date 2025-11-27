import time
import math
import matplotlib.pyplot as plt
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
import RPi.GPIO as GPIO
import matplotlib
# ------------------------------- GPIO -------------------------------
GPIO.setmode(GPIO.BCM)

TRIG = 23
ECHO = 24

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ------------------------------- SERVO ------------------------------
factory = LGPIOFactory()
servo = Servo(
    13,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory
)

def angle_to_servo_value(angle_deg):
    """
    Конвертация угла в диапазон [-1 .. 1] для gpiozero.Servo
    0° = -1
    90° = 0
    180° = +1
    """
    return (angle_deg - 90) / 90.0


# -------------------------- DISTANCE SENSOR -------------------------
def measure_distance():
    """Возвращает расстояние в см."""
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

    pulse_duration = pulse_end - pulse_start
    dist_cm = pulse_duration * 17150
    return round(dist_cm, 2)


# ----------------------------- MATPLOTLIB -----------------------------
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location("W")
ax.set_theta_direction(-1)
ax.set_rmax(75)     # максимум 2 метра
ax.set_title("Real-time Ultrasonic Scan")

# пустые точки
angles_plot = []
dist_plot = []
plot, = ax.plot([], [], 'ro')


# ----------------------------- MAIN LOOP ------------------------------
SCAN_MIN = 0
SCAN_MAX = 180
STEP = 5

print("Start scanning...")

while True:
    angles_plot.clear()
    dist_plot.clear()

    # скан LEFT → RIGHT
    for angle in range(SCAN_MIN, SCAN_MAX + 1, STEP):
        servo.value = angle_to_servo_value(angle)
        time.sleep(0.03)

        dist = measure_distance()
        print(f"Angle {angle}° → {dist} cm")

        # полярная матплотлиб работает в радианах
        angles_plot.append(math.radians(angle))
        dist_plot.append(dist)

        plot.set_data(angles_plot, dist_plot)
        plt.pause(0.0001)

    # скан RIGHT → LEFT (чтобы получать картнку чаще)
    for angle in range(SCAN_MAX, SCAN_MIN - 1, -STEP):
        servo.value = angle_to_servo_value(angle)
        time.sleep(0.03)

        dist = measure_distance()
        print(f"Angle {angle}° → {dist} cm")

        angles_plot.append(math.radians(angle))
        dist_plot.append(dist)

        plot.set_data(angles_plot, dist_plot)
        plt.pause(0.0001)
