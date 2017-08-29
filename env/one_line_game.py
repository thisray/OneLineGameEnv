## mimic ice-game environment: 1. one-line game
## build on python3

## TODO: add 'gym'

from __future__ import division
import numpy as np
import sys

randint = np.random.randint

"""
    OneLineGameEnv:

        - canvas_0:
            background: 0.0
            road: 0.5
            destination: 1.0
            (need init_position token ?)

        - canvas_1:
            background: 0.0
            now_position: 1.0
            past_path: 0.5

        - action_list = [0, 1, 2, 3, 4]

        - only canvas_1 would be changed

"""

class OneLineGameEnv():
    def __init__(self, L=8, worker_id=None):
        self.L = L
        self.N = self.L ** 2
        self.worker_id = worker_id

        lower_ratio, upper_ratio = 1.4, 2.0
        self.lower_length = 4 #int(self.L/lower_ratio)
        self.upper_length = 6 #int(self.L*upper_ratio)

        print('Build OneLineGameEnv() at worker_id:{} Success! length: {} to {}.'.format(self.worker_id, self.lower_length, self.upper_length))

        # canvas token setting
        self.background_token = 0.0
        self.road_token = 0.5
        self.destination_token = 1.0
        self.now_position_token = 1.0
        self.past_path_token = 0.5

        self.name_mapping = dict({
                                  0 :   'right',
                                  1 :   'down',
                                  2 :   'left',
                                  3 :   'up',
                                  4 :   'flag'
                                  ## 'lower_next', 'upper_next' in the future
                                  })

        self.index_mapping = dict({
                                  'right': 0,
                                  'down' : 1,
                                  'left' : 2,
                                  'up' : 3,
                                  'flag' : 4
                                  })

        # for the canvas
        self.init_position = (0, 0)
        self.now_position = (0, 0)
        self.position_list = []
        self.canvas_0 = np.zeros((self.L, self.L), dtype=np.float32)
        self.canvas_1 = np.zeros((self.L, self.L), dtype=np.float32)

        # other setting
        self.remain_life_upperbound = 4
        self.remain_life_now = self.remain_life_upperbound
        self.accepted_times = 0
        self.terminate_times = 0
        self.gameover_counter = 0
        self.acc_before_gameover = 0
        self.action_list = [0, 0, 0, 0, 0]


    def start(self, init_site=None):
        self.build_new_env(init_site)


    def reset(self, renew=False):
        if self.remain_life_now > 0 and renew == False:
            self.only_reset_canvas_1()
            self.remain_life_now = self.remain_life_now -1
        else:   
            # game over
            self.start()
            self.remain_life_now = self.remain_life_upperbound
            self.gameover_counter = self.gameover_counter +1
            count_game = 1000
            if self.gameover_counter % count_game == 0:
                action_static = [mem/sum(self.action_list) for mem in self.action_list]
                print('work_id: {}, now {} games, accepted {} times (in past {} games) ac: {}'.format(
                                    self.worker_id, self.gameover_counter, self.acc_before_gameover, count_game,
                                    [format(member, '.3f') for member in action_static]))

                self.acc_before_gameover = 0

        return self.get_obs()


    ## ===================================== ##

    def step(self, action, show=False):
        flag_action = self.index_mapping['flag']
        self.action_list[action] = self.action_list[action] +1
        terminate = False
        reward = 0.0
        rets = None

        if 0 <= action < flag_action:
            terminate = self.walk_and_check(action)
        elif action == flag_action:
            terminate = True
            reward = self.flag_here()

        if terminate == True:
            self.terminate_times = self.terminate_times +1

        obs = self.get_obs()
        if show == True:
            print('after action "{}", reward: {}, terminate: {}'.format(action, reward, terminate))
        return obs, reward, terminate, rets


    def flag_here(self):
        now_x, now_y = self.now_position
        reward = 0.0
        # reward = -0.1  # with panelty
        if self.canvas_0[now_x][now_y] == self.destination_token:
            reward = 1.0
            self.accepted_times = self.accepted_times +1
            self.acc_before_gameover = self.acc_before_gameover +1
            self.remain_life_now = 0
            # print('Accepted! worker_id: {}, accepted_times: {}, terminate_times: {}, ratio: {}'.format(self.worker_id, 
                    # self.accepted_times, self.terminate_times, format((self.accepted_times/self.terminate_times), '.3f')))
        return reward

    ## return terminate = True or False
    def walk_and_check(self, action):
        accepted_walk = True
        new_x, new_y = self.walk(self.now_position, action)

        if self.canvas_0[new_x][new_y] == self.background_token:
            accepted_walk = False

        # walk on self.canvas_1
        self.now_position = (new_x, new_y)
        self.walk_on_canvas_1(self.now_position)

        # append whether walk accepted or not
        self.position_list.append((new_x, new_y))   
        return (not accepted_walk)


    def walk_on_canvas_1(self, position):
        past_x, past_y = self.position_list[-1]
        new_x, new_y = position
        self.canvas_1[past_x][past_y] = self.past_path_token
        self.canvas_1[new_x][new_y] = self.now_position_token


    def for_debug(self, c0 = True, c1 = True):
        if c0 == True:
            print('self.canvas_0: \n', self.canvas_0)
        if c1 == True:
            print('self.canvas_1: \n', self.canvas_1)
        # for i in range(len(self.canvas_0)):
        #     print(self.canvas_0[i], '\t', self.canvas_1[i], '\n')


    ## ===================================== ##


    def get_obs(self):
        return np.stack((self.canvas_0, self.canvas_1), axis=2)


    def build_new_env(self, init_position=None):
        if init_position == None or init_position >= self.L:
            self.init_position = (randint(self.L), randint(self.L))            

        # decide one_line
        road_length = randint(self.lower_length, self.upper_length)
        # road_length = 3

        road_direction_list = self.build_new_road(road_length)  # it's action list
        road_path_list = self.walk_road_path(self.init_position, road_direction_list)
        # print('road_length: ', road_length)
        # print('road_direction_list: ', road_direction_list)
        # print('road_path_list: ', road_path_list)

        # build self.canvas_0
        self.draw_canvas_0(road_path_list)

        # build self.canvas_1
        self.only_reset_canvas_1()


    def build_new_road(self, road_length):  # this function decides the difficult level (maybe cross)
        road_direction = []
        last_road_action = -1
        upper_bound_action = self.index_mapping['flag'] -1
        repeat_times = 2
        road_length = int(road_length / repeat_times)
        for i in range(road_length):
            next_road_action = randint(0, upper_bound_action)   # only 0~4
            # while next_road_action == last_road_action:
            while self.go_back_check(next_road_action, last_road_action):
                next_road_action = randint(0, upper_bound_action)
            for j in range(repeat_times):
                road_direction.append(next_road_action)
            last_road_action = next_road_action
        return road_direction

    def go_back_check(self, action_1, action_2):
        check_result = False     # True: Not ok
        # (0, 2), (1, 3)
        if (action_1 == 0 and action_2 == 2) or \
           (action_1 == 2 and action_2 == 0) or \
           (action_1 == 1 and action_2 == 3) or \
           (action_1 == 3 and action_2 == 1):
            check_result = True
        return check_result

    def walk_road_path(self, init_position, road_direction_list):
        now_position = init_position
        road_path_list = [init_position]
        for i in range(len(road_direction_list)):
            now_position = self.walk(now_position, road_direction_list[i])
            road_path_list.append(now_position)
        return road_path_list

    def draw_canvas_0(self, road_path_list):
        self.canvas_0 = np.zeros((self.L, self.L), dtype=np.float32)
        # draw the road
        for i in range(len(road_path_list)):
            _x, _y = road_path_list[i]
            self.canvas_0[_x][_y] = self.road_token
        # draw the destination
        _x, _y = road_path_list[-1]
        self.canvas_0[_x][_y] = self.destination_token

    def only_reset_canvas_1(self):  # while self.remain_life_now > 0
        self.canvas_1 = np.zeros((self.L, self.L), dtype=np.float32)
        self.canvas_1[self.init_position[0]][self.init_position[1]] = self.now_position_token
        self.now_position = self.init_position  # reset position
        self.position_list = [self.init_position]     # reset position_list




    ## input: position & action, output: new_position (all 2D)
    def walk(self, position, action):
        """    
            0 :   'right',
            1 :   'down',
            2 :   'left',
            3 :   'up',
        """
        (x, y) = position
        x_p = x + 1 if x < self.L-1 else 0
        x_m = x - 1 if x > 0        else self.L-1
        y_p = y + 1 if y < self.L-1 else 0
        y_m = y - 1 if y > 0        else self.L-1

        new_position = (0, 0)

        if action == 0:
            new_position = (x, y_p) # down: (x_p, y)
        elif action == 1:
            new_position = (x_p, y) # left: (x, y_m)
        elif action == 2:
            new_position = (x, y_m) # up: (x_m, y)
        elif action == 3:
            new_position = (x_m, y) # right: (x, y_p)
        return new_position







