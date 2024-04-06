# from environments.env_wrapper import *
# from utils.commons import *
#
#
# def agent_test(agent: AgentOnline, env: EnvWrapper, test_episodes: int, logger: Logger, **kwargs):
#     reward_array = np.zeros(test_episodes)
#     for episode_i in range(test_episodes):
#         state = env.reset()
#         done = False
#         while not done:
#             action = agent.react(state, **kwargs)
#             next_state, reward, done, _ = env.step(action)
#             state = next_state
#             reward_array[episode_i] += reward
#             # if done:
#     logger(f"agent_test:\tTest Episodes Number: {test_episodes}, Total Reward: {np.mean(reward_array)}")
#     # todo
#     # tensorboard
#
#
# def online_training(agent: AgentOnline, env: EnvWrapper, env_test: EnvWrapper,
#                     num_episodes: int, test_interval: int, save_interval: int, logger: Logger, **kwargs):
#     training_steps = 0
#     for episode_i in range(num_episodes):
#         state = env.reset()
#         done = False
#         while not done:
#             action = agent.react(state, **kwargs)
#             next_state, reward, done, truncated, info = env.step(action)
#             agent.observe(transition=[next_state, reward, done, truncated, info])
#             agent.learn(**kwargs)
#             state = next_state
#
#         if episode_i % test_interval == 0:
#             agent_test(agent, env_test)
#
