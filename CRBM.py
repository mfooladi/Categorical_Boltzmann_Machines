import numpy as np
from mayavi.mlab import *


np.set_printoptions(linewidth=600)
np.set_printoptions(precision=5, edgeitems=25)


class CRBM:
    def __init__(self, weights):
        self.weights = weights
        self.hidden_units = len(self.weights[0])
        self.num_visible = len(self.weights)
        self.visible_states = list()

        for i in range(self.num_visible):
            self.visible_states.append(int(np.size(weights[i]) / self.hidden_units))

        vs_copy = self.visible_states.copy()
        vs_copy.insert(0, self.hidden_units)

        self.probs_shape = tuple(vs_copy)
        self.probs = np.zeros(self.probs_shape)

    def _net_input(self, states):
        print('states')
        print(states)
        e_net_sum = 0
        # partition_func_sum = 0
        e_partition_func = 0

        for i in range(self.num_visible):
            e_net_sum += self.weights[i][states[0]][states[i + 1]]

        e_net = np.exp(e_net_sum)

        for j in range(self.hidden_units):
            partition_func_sum = 0
            for k in range(self.num_visible):
                partition_func_sum += self.weights[k][j][states[k + 1]]

            e_partition_func += np.exp(partition_func_sum)

        prob_ = e_net / e_partition_func

        return prob_

    def _a_ijk(self):
        for sj in range(5):
            for si in range(5):
                for ul in range(7):
                    input_states = [sj, si, ul]
                    self.probs[sj, si, ul] = self._net_input(input_states)

        return self.probs


if __name__ == '__main__':
    w1 = np.array([[3.0, 2.9, 2.8, 2.7, 2.6],
                   [2.9, 3.0, 2.9, 2.8, 2.7],
                   [2.8, 2.9, 3.0, 2.9, 2.8],
                   [2.7, 2.8, 2.9, 3.0, 2.9],
                   [2.6, 2.7, 2.8, 2.9, 3.0]])

    w2 = ([[3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4],
           [2.9, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5],
           [2.8, 2.9, 3.0, 2.9, 2.8, 2.7, 2.6],
           [2.7, 2.8, 2.9, 3.0, 2.9, 2.8, 2.7],
           [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]])

    ws = [w1, w2]

    r = CRBM(weights=ws)

    probs = r._a_ijk()

    for i in range(5):
        figure(i)
        barchart(probs[i, :, :])

    show()



