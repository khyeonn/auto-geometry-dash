import cv2 as cv
import numpy as np
import pynput.mouse as ms
import pynput.keyboard as kb
from windowcapture import WindowCapture
from imageprocessor import ImageProcessor
import time
from collections import deque

from ppo_torch import Agent

keyboard = kb.Controller()
mouse = ms.Controller()

window_name = "Geometry Dash"
cfg_file_name = "yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_final.weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

def press_space():
    keyboard.press(kb.Key.space)
    keyboard.release(kb.Key.space)

def press_esc():
    keyboard.press(kb.Key.esc)
    keyboard.release(kb.Key.esc)

def retry():
    mouse.position = (1600, 800)
    mouse.press(ms.Button.left)
    mouse.release(ms.Button.left)

def preprocess(state, game_time):
    # width = img.shape[1]
    # height = img.shape[0]
    # dim = (abs(width/2), abs(height/2))
    resized = cv.resize(state, (80, 105)) #interpolation = cv2.INTER_AREA)
    resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    resized = resized/255.0 # convert all pixel values in [0,1] range

    max_game_time = 2500.0
    normalized_game_time = game_time/max_game_time

    game_time_channel = np.full((resized.shape[0], resized.shape[1], 1), normalized_game_time)

    state_with_time = np.dstack((resized, game_time_channel))

    # resized = resized.reshape(resized.shape + (1,))
    return state_with_time

ss = wincap.get_screenshot()
ss_resized = ss[1:1000, 620:1400]
input_dims = preprocess(ss_resized, 0).flatten().shape

# countdown 
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)
retry()


## PPO hyperparameters
buffer_len = 2048
batch_size = 1024
n_epochs = 20
alpha = 1e-3
policy_clip = 0.2
agent = Agent(n_actions=2, batch_size=batch_size, buffer_len=buffer_len, policy_clip=policy_clip,
              alpha=alpha, n_epochs=n_epochs, input_dims=input_dims)


score = 0
old_best_score = 0
best_score = 0
avg_score = 0
score_history = []
n_attempts = 1
update_interval = 10
maxlen = 5


# timing
prev_time = time.time()
elapsed_time = 0
total_time = 0
alive_time = 0
game_time = 0

done = False
just_died = True
buffer = deque(maxlen=maxlen)
while(total_time < 85):    
    elapsed_time += time.time() - prev_time
    total_time += time.time() - prev_time
    prev_time = time.time()
    if elapsed_time > 0.016:
        elapsed_time = 0

        ss = wincap.get_screenshot()
        
        # cv.imshow('a', ss[1:1000, 620:1400])

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

        isAlive = improc.proccess_image(ss)

        if isAlive:
            alive_time += 1 
        else:
            alive_time -= 1
        
        if alive_time > 10:
            done = False
            alive_time = 0
            just_died = False
        elif alive_time < -10:
            done = True
            alive_time = 0

        if not done:
            reward = 0.1
            game_time += 1

            observation = preprocess(ss[1:1000, 620:1400], game_time)
            action, prob, val = agent.choose_action(observation.flatten())

            # # print(action, prob, val)
            if action == 1:
                press_space() 

            score += reward

            agent.remember(observation.flatten(), action, prob, val, reward, done)
            buffer.append((observation, action, prob, val, reward, done, ss))
        else:
            if not just_died:
                just_died = True
                game_time = 0
                total_time = 0
                for _ in range(maxlen):
                    agent.forget()

                for i in range(maxlen):
                    last_observation, last_action, last_prob, last_val, _, _, ss = buffer[i]
                    # cv.imshow('a', ss)
                    agent.remember(last_observation.flatten(), last_action, last_prob, last_val, 0, True)
                    

                buffer.clear()
                score_history.append(score)
                best_score = np.max(score_history)
                avg_score = np.mean(score_history[-100:])
                n_attempts += 1
                score = 0 

                if best_score > old_best_score:
                    old_best_score = best_score
                    agent.save_models()

        if n_attempts % update_interval == 0:
            n_attempts += 1
            print('... learning ...')
            press_esc()
            agent.learn()
            retry()

            # reset variables
            game_time = 0
            total_time = 0
            alive_time = 0
            score = 0
            buffer.clear()
            just_died = True

        # if game_time % 10 == 0 and game_time != 0:
        #     print('attempt', n_attempts, 'score %.1f' % score, 'best score %.1f' % old_best_score, 'avg score %.1f' % avg_score)

