import cv2
import numpy as np
import random
import copy


class Tetris:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.back_ground = np.zeros([self.h, self.w])
        self.action_space = ['gl', 'gr', 'gd']
        self.step_num = 0
        # https://en.wikipedia.org/wiki/Tetromino
        self.tetrominoes = {
            # O
            0: {0: [(0, 1), (1, 1), (0, 0), (1, 0)],
                90: [(0, 1), (1, 1), (0, 0), (1, 0)],
                180: [(0, 1), (1, 1), (0, 0), (1, 0)],
                270: [(0, 1), (1, 1), (0, 0), (1, 0)]},
            #'I'
            1: {0: [(0, 1), (0, 0), (0, -1), (0, -2)],
                90: [(-1, 0), (0, 0), (1, 0), (2, 0)],
                180: [(0, 1), (0, 0), (0, -1), (0, -2)],
                270: [(-1, 0), (0, 0), (1, 0), (2, 0)]
                }
            }
        self.current_tetrominoes_position = [0, 0]
        self.current_tetrominoes_type = random.choice(list(self.tetrominoes.keys()))
        self.current_tetrominoes_angle = random.choice([90, 180, 270, 0])

        pass

    def reset(self):
        self.current_tetrominoes_type = random.choice(list(self.tetrominoes.keys()))
        self.step_num = 0
        self.current_tetrominoes_angle = random.choice([90, 180, 270, 0])
        self.back_ground = np.zeros([self.h, self.w])
        self.current_tetrominoes_position = [int(self.w / 2), self.h]
        return np.append(np.append(self.back_ground, [self.current_tetrominoes_type, self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32)

    def _test_move(self, new_position):
        for pos in self.tetrominoes[self.current_tetrominoes_type][self.current_tetrominoes_angle]:
            x = pos[0] + new_position[0]
            y = pos[1] + new_position[1]
            if x < 0 or y < 0 or x >= self.w:
                return False
            if y < self.h:
                if self.back_ground[y][x] != 0:
                    return False
        return True

    def _test_end(self, new_position):
        for pos in self.tetrominoes[self.current_tetrominoes_type][self.current_tetrominoes_angle]:
            x = pos[0] + new_position[0]
            y = pos[1] + new_position[1]
            if y >= self.h:
                return True

    def _test_angle(self, new_angle):
        for pos in self.tetrominoes[self.current_tetrominoes_type][new_angle]:
            x = pos[0] + self.current_tetrominoes_position[0]
            y = pos[1] + self.current_tetrominoes_position[1]
            if x < 0 or y < 0 or x >= self.w:
                return False
            if y < self.h:
                if self.back_ground[y][x] != 0:
                    return False
        return True

    def _update_background(self):

        reward = 0
        line = 0
        for pos in self.tetrominoes[self.current_tetrominoes_type][self.current_tetrominoes_angle]:
            x = pos[0] + self.current_tetrominoes_position[0]
            y = pos[1] + self.current_tetrominoes_position[1]
            if 0 <= x < self.w and 0 <= y < self.h:
                self.back_ground[y][x] = 1
        sum_of_each_row = np.sum(self.back_ground, axis=1)
        full_row = []
        for i in range(len(sum_of_each_row)):
            if sum_of_each_row[i] == self.w:
                reward += 1
                line += 1
                full_row.append(i)
        self.back_ground = np.delete(self.back_ground, full_row, axis=0)
        # print(self.back_ground)
        if line != 0:
            new_rows = np.zeros((line, self.w))
            self.back_ground = np.row_stack((self.back_ground, new_rows))
        return line, reward

    def _new_tetrominoes(self):
        self.current_tetrominoes_type = random.choice(list(self.tetrominoes.keys()))
        self.current_tetrominoes_angle = random.choice([90, 180, 270, 0])
        self.current_tetrominoes_position = [int(self.w / 2), self.h]

    def step(self, action):
        reward_0 = 0
        if self.step_num % 4 == 0:
            state, reward_0, is_done, _ = self.step_raw(2)
            if is_done:
                return state, reward_0, is_done, _
        state, reward, is_done, _ = self.step_raw(action)
        return state, reward+reward_0, is_done, _

    def step_raw(self, action):
        self.step_num += 1
        if self.action_space[action] == 'gl':
            new_position = [self.current_tetrominoes_position[0] - 1,
                            self.current_tetrominoes_position[1]]
            if self._test_move(new_position):
                self.current_tetrominoes_position = new_position
            return np.append(np.append(self.back_ground, [self.current_tetrominoes_type,
                                                          self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32), 0, False, {}
        elif self.action_space[action] == 'gr':
            new_position = [self.current_tetrominoes_position[0] + 1,
                            self.current_tetrominoes_position[1]]
            if self._test_move(new_position):
                self.current_tetrominoes_position = new_position
            return np.append(np.append(self.back_ground, [self.current_tetrominoes_type,
                                                          self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32), 0, False, {}
        elif self.action_space[action] == 'gd':
            new_position = [self.current_tetrominoes_position[0],
                            self.current_tetrominoes_position[1] - 1]
            if self._test_move(new_position):
                self.current_tetrominoes_position = new_position
                return np.append(np.append(self.back_ground, [self.current_tetrominoes_type,
                                                              self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32), 0, False, {}
            else:
                if self._test_end(new_position):
                    return np.append(np.append(self.back_ground, [self.current_tetrominoes_type,
                                                                  self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32), 0, True, {}
                line, reward = self._update_background()
                self._new_tetrominoes()
                return np.append(np.append(self.back_ground, [self.current_tetrominoes_type,
                                                              self.current_tetrominoes_angle]),
                         self.current_tetrominoes_position).astype(np.float32), reward, False, {}

    def draw(self, resize_factor=100):
        back_ground = self.back_ground
        tetrom_type = self.current_tetrominoes_type
        tetrom_angle = self.current_tetrominoes_angle
        tetrom_position = self.current_tetrominoes_position
        gray = np.zeros([self.h, self.w, 1], np.uint8)
        bg = copy.deepcopy(self.back_ground).astype(np.uint8) * 255
        screen = cv2.merge([gray, bg, gray])
        for pos in self.tetrominoes[tetrom_type][tetrom_angle]:
            x = pos[0] + tetrom_position[0]
            y = pos[1] + tetrom_position[1]
            if x < 0 or y < 0 or x >= self.w or y >= self.h:
                continue
            else:
                screen[y][x] = [255, 0, 0]
        frame = cv2.flip(
            cv2.resize(screen, (self.w * resize_factor, self.h * resize_factor), interpolation=cv2.INTER_NEAREST), 0)
        return frame


if __name__ == '__main__':
    t = Tetris(10, 16)
    t.reset()
    is_done = False
    while not is_done:
        frame = t.draw()
        cv2.imshow('play', frame)
        action = cv2.waitKey(2000)
        if action == ord('w'):
            action = 3
        elif action == ord('a'):
            action = 0
        elif action == ord('s'):
            action = 2
        elif action == ord('d'):
            action = 1
        else:
           continue
        state, reward, is_done, _ = t.step(int(action))
        print(reward)
