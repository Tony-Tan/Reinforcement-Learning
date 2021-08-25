import numpy as np

SMALL_FLOUT_NUM = 1e-10


class UCB:
    def __init__(self, c):
        self._c = c

    def __call__(self, action_value_array, current_step_num, actions_selected_times_array):
        action_space_n = len(action_value_array)
        prob = np.zeros(action_space_n)
        actions_selected_times_array = np.array(actions_selected_times_array) + SMALL_FLOUT_NUM
        uncertainty_array = \
            action_value_array + self._c * np.sqrt(np.log(current_step_num) / actions_selected_times_array)
        action_selected = np.random.choice(np.flatnonzero(uncertainty_array == uncertainty_array.max()))
        prob[action_selected] = 1.0
        return prob
