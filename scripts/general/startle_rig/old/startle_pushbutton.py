import time
from time import sleep
from time import strftime
import RPi.GPIO as GPIO
from datetime import datetime
import socket
import subprocess
from gpiozero import Device
from gpiozero.pins.rpigpio import RPiGPIOFactory

from gpiozero import LED
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder

Device.pin_factory = RPiGPIOFactory()

###################
#### LED SETUP ####
###################
ir = LED(5)
ir.off()

white_led = LED(6)
white_led.off()

###################
#### PIN SETUP ####
###################
shutdown_pin = 2
record_pin = 27
stop_pin = 22
red_led = 23
yellow_led = 24

# Suppress warnings
GPIO.setwarnings(False)

# Use "GPIO" pin numbering
GPIO.setmode(GPIO.BCM)

# Set up GPIO input pins with internal pull-up resistors (Push buttons)
GPIO.setup(shutdown_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(record_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set up GPIO output pins (LED's)
GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(yellow_led, GPIO.OUT)

# Start with both LED's off
GPIO.output(red_led,GPIO.LOW)
GPIO.output(yellow_led,GPIO.LOW)

#######################
#### CAMERA SET UP ####
#######################
rig_name = socket.getfqdn()
camera = Picamera2()
camera.resolution = (1100, 1800)
camera.framerate = 5
camera.rotation = 90
encoder = H264Encoder(bitrate=10_000_000)


#######################
#### SHUTDOWN LED  ####
#######################
def shut_down():
    GPIO.output(yellow_led, GPIO.HIGH) # Turn on yellow LED to indicate shutdown process
    print("shutting down")
    command = "/usr/bin/sudo /sbin/shutdown -h now"
    try:
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error shutting down: {e}")

##############################
#### EXPERIMENT INTERVALS ####
##############################

initial_interval = 5 
n_exp = 2
flash_interval = 2
habituation = 10


try:
    while True:
         #short delay, otherwise this code will take up a lot of the Pi's processing power
        time.sleep(0.2)

        if GPIO.input(record_pin) == False:
            GPIO.output(red_led,GPIO.HIGH) # Red LED ON to indicate the device is recording
            camera.start_preview(Preview.QTGL)
            camera.start()

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            camera.start_recording(encoder, f"/home/topdown/data/{date}_{rig_name}.h264")
            ir.on()

            try:
                for i in range(n_exp):

                    # ir.on()
                    sleep(initial_interval)

                    white_led.on()
                    sleep(flash_interval)
                    white_led.off()
                    
                    sleep(habituation)

                    # ir.off()

                    # ---- STOP BUTTON GOES HERE ----
                    if GPIO.input(stop_pin) == False:
                        time.sleep(0.2)                 # debounce
                        if GPIO.input(stop_pin) == False:
                            break
           
            finally:
                camera.stop_recording()                 # <--- stop recording when loop ends or stop pressed
                camera.stop_preview()
                GPIO.output(red_led, GPIO.LOW) # Red LED OFF to indicate the recording stopped
                ir.off()
                white_led.off()

finally:
    GPIO.cleanup()