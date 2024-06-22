from agents.dqn_agent import *
from abc_rl.experience_replay import *
from abc_rl.exploration import *
from utils.hyperparameters import *


class DQNPlayGround:
    def __init__(self, agent: DQNAgent, env: EnvWrapper, cfg: Hyperparameters, logger: Logger):
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.logger = logger

    def train(self):
        # training
        epoch_i = 0
        training_steps = 0
        while training_steps < self.cfg['training_steps']:
            state, inf = self.env.reset()
            if 'lives' in inf.keys():
                last_lives = inf['lives']
            else:
                last_lives = None
            done = False
            truncated = False
            step_i = 0
            run_test = False
            reward_cumulated = 0
            obs = self.agent.perception_mapping(state, step_i)
            while (not done) and (not truncated):
                # no op for the first few steps and then select action by epsilon greedy or other exploration methods
                if len(self.agent.memory) > self.cfg['replay_start_size'] and step_i >= self.cfg['no_op']:
                    action = self.agent.select_action(obs)
                else:
                    action = self.agent.select_action(obs, RandomAction())
                next_state, reward_raw, done, truncated, inf = self.env.step(action)
                if 'lives' in inf.keys():
                    current_lives = inf['lives']
                    reward = self.agent.reward_shaping(reward_raw, last_lives, current_lives)
                    last_lives = current_lives
                else:
                    reward = self.agent.reward_shaping(reward_raw)
                next_obs = self.agent.perception_mapping(next_state, step_i)
                self.agent.store(obs, action, reward, next_obs, done, truncated)
                self.agent.train_step()
                obs = next_obs
                reward_cumulated += reward_raw

                if (len(self.agent.memory) > self.cfg['replay_start_size'] and
                        training_steps % self.cfg['batch_num_per_epoch'] == 0):
                    run_test = True
                    epoch_i += 1
                training_steps += 1
                step_i += 1
            self.logger.tb_scalar('training reward', reward_cumulated, training_steps)
            if run_test:
                self.logger.msg(f'{epoch_i} test start:')
                avg_reward, avg_steps = self.test(self.cfg['agent_test_episodes'])
                self.logger.tb_scalar('avg_reward', avg_reward, epoch_i)
                self.logger.tb_scalar('avg_steps', avg_steps, epoch_i)
                self.logger.tb_scalar('epsilon', self.agent.exploration_method.epsilon, epoch_i)
                self.logger.msg(f'{epoch_i} avg_reward: ' + str(avg_reward))
                self.logger.msg(f'{epoch_i} avg_steps: ' + str(avg_steps))
                self.logger.msg(f'{epoch_i} epsilon: ' + str(self.agent.exploration_method.epsilon))

    def test(self, test_episode_num: int):
        """
        Test the DQN agent for a given number of episodes.
        :param test_episode_num: The number of episodes for testing
        :return: The average reward and average steps per episode
        """
        env = EnvWrapper(self.cfg['env_name'], repeat_action_probability=0, frameskip=self.cfg['skip_k_frame'])
        exploration_method = EpsilonGreedy(self.cfg['epsilon_for_test'])
        reward_cum = 0
        step_cum = 0
        for i in range(test_episode_num):
            state, _ = env.reset()
            done = truncated = False
            step_i = 0
            while (not done) and (not truncated):
                obs = self.agent.perception_mapping(state, step_i)
                action = self.agent.select_action(obs, exploration_method)
                next_state, reward, done, truncated, inf = env.step(action)
                reward_cum += reward
                state = next_state
                step_i += 1
            step_cum += step_i
        return reward_cum / self.cfg['agent_test_episodes'], step_cum / self.cfg['agent_test_episodes']
