from gpiozero import LED
from time import sleep
from picamera2 import Picamera2, Preview
from datetime import datetime
import socket


## RIG NAME
rig_name = socket.getfqdn()

## TURN THE LED LIGHT ON 
ir = LED(5, active_high=False)
ir.on()

## WHITE LED
white_led = LED(6)
white_led.off()

## SET UP THE CAMERA PREVIEW
camera = Picamera2()
preview_config = camera.create_preview_configuration(main={"size": (1800, 1100)})
camera.configure(preview_config)
camera.start_preview(Preview.QTGL)
camera.start()


interval = 300
number_exp = 3
flash_duration = 1  # seconds


date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_path = f"/home/sv18/data/{date}_{rig_name}.h264"

camera.start_recording(video_path)



ir.on()
sleep(60)

try:
    for i in range(number_exp):
                
        white_led.on()
        sleep(flash_duration)
        white_led.off()
        sleep(interval)
     
        # ir.off() 

except KeyboardInterrupt:
    print("Interrupted by the user")
    
finally:
    camera.stop_recording()
    camera.stop_preview()
    camera.stop()
    #Switch led's off
    white_led.off()
    ir.off()

print("Experiment completed")

sleep(999)