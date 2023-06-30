from __future__ import division
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class synth_expert:
    def __init__(self, flip_prob=0.30, p_in=0.70):
        self.n_classes = 2
        self.flip_prob = flip_prob
        self.p_in = p_in	

    def predict_prob(self, input, labels, hpred):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            coin_flip = np.random.binomial(1, self.p_in)
            if coin_flip == 1:
                # we make use of hpred to make sure that we get the actual correct expert decision and not the flipped y
                outs[i] = hpred[i].item()
            if coin_flip == 0:
                outs[i] = random.randint(0, self.n_classes - 1)
        return outs
