from gpiozero import Device
from gpiozero.pins.rpigpio import RPiGPIOFactory
Device.pin_factory = RPiGPIOFactory()

from gpiozero import LED
from time import sleep
from picamera2 import Picamera2, Preview


ir = LED(5)
ir.off()

white_led = LED(6)
white_led.off()


camera = Picamera2()

preview_config = camera.create_preview_configuration(main={"size": (1000, 1000)})
camera.configure(preview_config)

camera.start_preview(Preview.QTGL)
camera.start()


initial_interval = 5 #15 minutes * 60 seconds
n_exp = 2
flash_interval = 2
habituation = 10


#Here you would start recording
try:
    for i in range(n_exp):
                
        ir.on()
        sleep(initial_interval)
        
        white_led.on()
        sleep(flash_interval)
        white_led.off()
        
        sleep(habituation)
     
        ir.off()

    
except KeyboardInterrupt:
    print("Interrupted by the user")
    
finally:
    ir.off()
    white_led.off()

print("Experiment completed")

