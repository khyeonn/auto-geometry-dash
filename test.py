import cv2 as cv
import numpy as np
import mss
from time import time, sleep
from pynput.keyboard import Key, Controller
from neuralnetwork import NeuralNetwork


import torch
import torch.nn.functional as F
import torch.optim as optim

keyboard = Controller()

template = cv.imread('attempt.jpg')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

retry_template = cv.imread('player.jpg')
retry_template = cv.cvtColor(retry_template, cv.COLOR_BGR2GRAY)

reward = 0.0
new_attempt = False

attempt_states = []
attempt_actions = []
attempt_rewards = []


def select_action(img, policy_net):
    with torch.no_grad():
        state = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        logits = policy_net(state)
        action_probs = F.softmax(logits, dim=1)
        action = torch.multinomial(action_probs, num_samples=1).item()
    return action, logits

def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32)
    running_add = 0
    for r in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[r]
        discounted_rewards[r] = running_add
    return discounted_rewards


def update_policy(policy_net, states, actions, rewards, gamma=0.99):
    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    logits = [policy_net(state) for state in states]
    action_probs = [F.softmax(logit, dim=1) for logit in logits]

    chosen_action_probs = [torch.gather(action_prob, 1, action.unsqueeze(1)) for action_prob, action in zip(action_probs, actions)]
    log_probs = [torch.log(chosen_action_prob) for chosen_action_prob in chosen_action_probs]
    advantages = discounted_rewards - discounted_rewards.mean()
    policy_loss = -(torch.cat(log_probs) * advantages.detach()).mean()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


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
    img = np.array(sct.grab(monitor))
    img_nn = img[220:1024, 312:900]
    img_nn_gray = cv.cvtColor(img_nn, cv.COLOR_BGR2GRAY)
    _, img_nn_proc = cv.threshold(img_nn_gray, 128, 255, cv.THRESH_BINARY)
    input_shape = (img_nn_proc.shape[0], img_nn_proc.shape[1], 1)
    policy_net = NeuralNetwork(input_shape)

    retry = False
    new_attempt = False
    elapsed_time = 0
    count = 0

    while "Screen capturing":
        # get raw pixels from the screen, save it to a np array
        img = np.array(sct.grab(monitor))

        # region of interest for attempt detection
        img_cropped = img[100:250, 0:1000]

        # region of interest of retry button
        # img_retry = img[1:1200, 180:320]
        # hsv = cv.cvtColor(img_retry, cv.COLOR_BGR2HSV)
        # lower_green = np.array([35, 100, 100])
        # upper_green = np.array([110, 255, 255])
        # mask = cv.inRange(hsv, lower_green, upper_green)
        # green_pixels = cv.countNonZero(mask)

        # img_retry_gray = cv.cvtColor(img_retry, cv.COLOR_BGR2GRAY)
        # _, img_retry_proc = cv.threshold(img_retry_gray, 150, 255, cv.THRESH_BINARY)
        # black_pixels = cv.countNonZero(img_retry_proc)

        # ROI for input to neural net 
        img_nn = img[220:1024, 312:900]
        # preprocess image for NN
        img_nn_gray = cv.cvtColor(img_nn, cv.COLOR_BGR2GRAY)
        _, img_nn_proc = cv.threshold(img_nn_gray, 128, 255, cv.THRESH_BINARY)

        # more preprocessing
        img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
        _, img_proc = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY)

        # template matching for new attempt detection
        template_cropped = cv.resize(template, (img_cropped.shape[1], img_cropped.shape[0])) # template needs to have matching dimensions as image
        res = cv.matchTemplate(img_proc, template_cropped, cv.TM_CCOEFF_NORMED)
        threshold = 0.20
        loc = np.where(res >= threshold)

        # retry_templated_cropped = cv.resize(retry_template, (img_retry.shape[1], img_retry.shape[0])) # template needs to have matching dimensions as image
        # retry_res = cv.matchTemplate(img_retry_proc, retry_templated_cropped, cv.TM_CCOEFF_NORMED)
        # retry_threshold = 0.16
        # retry_loc = np.where(retry_res >= retry_threshold)
        # if green_pixels < 3000:
        #     retry = True

        if len(loc[0]) == 1:
            # print('new')
            new_attempt = True

        if new_attempt and len(loc[0]) == 1:
            count += 1

        print(count)
        if count >= 15:
            new_attempt = False
            retry = True

        
        # while new_attempt:
        if new_attempt:

            # last_time = time()

            # elapsed_time += time() - last_time
            # print('attempting')
            # if elapsed_time > 0.016:
            #     elapsed_time = 0                
                
                
            action, logits = select_action(img_nn_proc, policy_net)

            attempt_states.append(img_nn_proc)
            attempt_actions.append(action)
            attempt_rewards.append(reward)
            reward += 1.0    


            if action == 1:
                keyboard.press(Key.space)
                keyboard.release(Key.space)            

        if retry:
            if not attempt_states:
                # print('empty')                                                     
                pass
            else:
                # print(count)
                new_attempt = False
                retry = False
                count = 0
                # sleep(0.25)

                if len(attempt_states) > 512:
                    print('updating')
                    keyboard.press(Key.esc)
                    sleep(0.02)
                    keyboard.release(Key.esc)
                    update_policy(policy_net, attempt_states, attempt_actions, attempt_rewards, gamma=0.99)
                    print('updated')  

                    keyboard.press(Key.space)
                    sleep(0.02)
                    keyboard.release(Key.space) 

                    attempt_states = []
                    attempt_actions = []       
                    attempt_rewards = []

                reward = 0.0      

                print('resetting')
                # sleep(1)
                # keyboard.press(Key.space)
                # sleep(0.002)
                # keyboard.release(Key.space)
                # print('reset')    




        # show preprocessed image
        cv.imshow('geodash', img_proc)

        # press "q" to quit
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break

        