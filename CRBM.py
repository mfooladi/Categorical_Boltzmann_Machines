import numpy as np
from mayavi.mlab import *


np.set_printoptions(linewidth=600)
np.set_printoptions(precision=5, edgeitems=25)

num_hidden = np.array([5])
num_visible = np.array([5, 7])

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


class CRBM:
    def __init__(self, hidden_state, visible_state, weights):
        self.hidden_states = hidden_state
        self.visible_states = visible_state
        self.weights = weights
        self.probs = np.zeros((5, 5, 7))

    def _net_input(self, j, si, ul):
        partition_func = 0
        e_net = np.exp(self.weights[0][j][si] + self.weights[1][j][ul])

        for i in range(5):
            partition_func += np.exp(self.weights[0][i][si] + self.weights[1][i][ul])

        prob_jsiul = e_net / partition_func

        return prob_jsiul

    def _a_ijk(self):
        for sj in range(5):
            for si in range(5):
                for ul in range(7):
                    self.probs[sj, si, ul] = r._net_input(sj, si, ul)

        return self.probs


if __name__ == '__main__':
    r = CRBM(hidden_state=num_hidden, visible_state=num_visible, weights=ws)

    probs = r._a_ijk()

    for i in range(5):
        figure(i)
        barchart(probs[i, :, :])

    show()



