import dqn as dqn
import os
import time

if __name__ == '__main__':

    env_name = "ALE/Boxing-v5"
    env_ = dqn.DQNGym(env_name)
    agent = dqn.AgentDQN(env_.action_dim, os.path.join('./model/'))
    agent.load('./model/03-16-11-07-57_ALE_Boxing-v5/03-17-08-11-22_value.pth')
    agent.epsilon = 0.00001
    dqn_play_ground = dqn.DQNPlayGround(env_, agent)
    frame_num = 0
    last_record_episode = None
    frame_num_last_record = 0
    while True:
        dqn_play_ground.play_rounds(1000, display=True)
        frame_num += 1

