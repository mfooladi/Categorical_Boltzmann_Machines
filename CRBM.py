import numpy as np
from mayavi.mlab import *


np.set_printoptions(linewidth=600)
np.set_printoptions(precision=5, edgeitems=25)


class CRBM:
    def __init__(self, prob_name):
        self.w1 = np.array([[[3.0, 2.9, 2.8, 2.7, 2.6],
                             [2.9, 3.0, 2.9, 2.8, 2.7],
                   [2.8, 2.9, 3.0, 2.9, 2.8],
                   [2.7, 2.8, 2.9, 3.0, 2.9],
                   [2.6, 2.7, 2.8, 2.9, 3.0]],
                  [[3.0, 2.95, 2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55],
                   [2.9, 2.95, 3.0, 2.95, 2.9, 2.85, 2.8, 2.75, 2.7, 2.65],
                   [2.8, 2.85, 2.9, 2.95, 3.0, 2.95, 2.9, 2.85, 2.8, 2.75],
                   [2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 2.95, 2.9, 2.85],
                   [2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.0]]])

        self.w2 = np.array([[[2.6, 2.7, 2.8, 2.9, 3.0],
                             [3.0, 2.9, 2.8, 2.7, 2.6],
                             [2.4, 2.5, 2.6, 2.7, 2.8],
                             [2.4, 2.5, 2.6, 2.7, 2.8]],
                            [[2.6, 2.7, 2.8, 2.9, 3.0],
                             [2.4, 2.5, 2.6, 2.7, 2.8],
                             [3.0, 2.9, 2.8, 2.7, 2.6],
                             [2.4, 2.5, 2.6, 2.7, 2.8]],
                            [[2.6, 2.7, 2.8, 2.9, 3.0],
                             [2.4, 2.5, 2.6, 2.7, 2.8],
                             [2.4, 2.5, 2.6, 2.7, 2.8],
                             [3.0, 2.9, 2.8, 2.7, 2.6]]])

        if prob_name == 'transition':
            self.weights = self.w1
        elif prob_name == 'emission':
            self.weights = self.w2
        else:
            print('No probability name')

        self.hidden_units = len(self.weights[0])
        self.num_visible = len(self.weights)
        self.visible_states = list()

        for i in range(self.num_visible):
            self.visible_states.append(len(self.weights[i][0]))

        vs_copy = self.visible_states.copy()
        vs_copy.insert(0, self.hidden_units)

        self.probs_shape = tuple(vs_copy)
        self.probs = np.zeros(self.probs_shape)

        print('CRBM initialized')

    def _net_input(self, states):
        e_net_sum = 0
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
        self.probs = np.zeros(self.probs_shape)

        for sj in range(5):
            for si in range(5):
                for ul in range(10):
                    input_states = [sj, si, ul]
                    self.probs[sj, si, ul] = self._net_input(input_states)

        return np.transpose(self.probs)

    def total_transition_probs(self):
        p1 = self._a_ijk()
        p2 = self._a_ijk()
        p3 = self._a_ijk()
        total_trx_probs = np.kron(np.kron(p1, p2), p3)

        return total_trx_probs

    def _o_jk(self):
        self.probs = np.zeros(self.probs_shape)

        for yk in range(4):
            for s1 in range(5):
                for s2 in range(5):
                    for s3 in range(5):
                        input_states = [yk, s1, s2, s3]
                        self.probs[yk, s1, s2, s3] = self._net_input(input_states)
                        emission_reshape = self.probs.reshape(4, 125)

        return emission_reshape
        # return self.probs

if __name__ == '__main__':

    r = CRBM('transition')
    # r = CRBM('emission')

    probs = r._a_ijk()
    # probs = r._o_jk()

    for i in range(5):
        figure(i)
        barchart(r.probs[i])
        xlabel('S')
        ylabel('U')
        # zlabel('s3')

    show()



