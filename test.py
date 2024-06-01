import cv2 as cv
import numpy as np
from pynput.keyboard import Key, Controller
from windowcapture import WindowCapture
from imageprocessor import ImageProcessor
import time
import mss
from ppo_torch import Agent

keyboard = Controller()

window_name = "Geometry Dash"
cfg_file_name = "yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_final.weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

# while(True):
#     ss = wincap.get_screenshot()

#     if cv.waitKey(1) == ord('q'):
#         cv.destroyAllWindows()
#         break

#     player_states = improc.proccess_image(ss)
    
    
#     for state in player_states:
#         status = np.array([state["class_name"]])

#         # if status == 'alive':
#         #     keyboard.press(Key.space)
#         #     keyboard.release(Key.space)
        
#         # else:
#         #     pass
            

    
with mss.mss() as sct:
    # part of the screen to capture
    monitor_number = 1
    mon = sct.monitors[monitor_number]
    monitor = {
        "top": mon["top"] + 200,  
        "left": mon["left"] + 700, 
        "width": 1200,
        "height": 1200,
        "mon": monitor_number,
    }

    improc = ImageProcessor([1200, 1200], cfg_file_name, weights_file_name)

    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=2, batch_size=batch_size,
                alpha=alpha, n_epochs=n_epochs, input_dims=np.prod((1200, 1200)))

    while "Screen capturing":
        # get raw pixels from the screen, save it to a np array
        img = np.array(sct.grab(monitor))
        img = img[...,:3]
        img = np.ascontiguousarray(img) 

        coordinates = improc.proccess_image(img)

        # cv.imshow('geodash', img)

        action, prob, val = agent.choose_action(img.flatten())


        if cv.waitKey(1) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    break
            
    
        