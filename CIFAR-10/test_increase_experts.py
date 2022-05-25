import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from reliability_diagram import compute_calibration

n_classes = 10
confs = []
exps = []
true = []
path = "./validation/"
n_experts = [1, 2, 4, 6, 8]
for n in n_experts:
    model_name = '_' + str(n) + '_experts'
    with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
        conf = json.loads(json.load(f))
    with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
        exp_pred = json.loads(json.load(f))
    with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
        true_label = json.loads(json.load(f))
    true.append(true_label['test'])
    exps.append(exp_pred['test'])
    c = torch.tensor(conf['test'])
    c = F.softmax(c, dim=1)
    print(c.shape)
    temp = 0
    for i in range(n):
        temp += c[:, (n_classes + n) - (i + 1)]
    prob = c / (1.0 - temp).unsqueeze(-1)
    confs.append(prob)

ECEs = []
for i in range(len(n_experts)):
    print(i)
    c = confs[i]
    e = exps[i]
    t = torch.tensor(true[i])
    for j in range(len(e)):
        eces = []
        e_j = e[j]
        c_j = c[:, c.shape[1] - (j + 1)]
        t_j = t
        ids_where_gt_one = torch.where(c_j > 1.0)
        print(len(ids_where_gt_one[0]))
        c_j[ids_where_gt_one] = 1.0
        acc_j = t_j.eq(torch.tensor(e_j))
        log = compute_calibration(c_j, acc_j)
        eces.append(log['expected_calibration_error'])
    ECEs.append(eces)

Y = []
for l in ECEs:
    Y.append(np.average(l))

print(Y)
#plt.plot([y*100 for y in Y])
#plt.show()