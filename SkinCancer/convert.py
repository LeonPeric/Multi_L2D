import pickle5 as pickle
import json
"""
Reads all the invidual pickle files and combines them into one.
"""
noise_rates = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5]
seeds = [42, 35, 936, 235, 464, 912, 445, 202, 19, 986]

metrics_softmax = []
metrics_ova = []
for noise in noise_rates:
    noise_analytics_softmax = []
    noise_analytics_ova = []
    for seed in seeds:
        with open(f"metrics/softmax/metrics_softmax_{seed}_{noise}.pickle", "rb") as f:
            noise_analytics_softmax.append(pickle.load(f))
        with open(f"metrics/ova/metrics_ova_{seed}_{noise}.pickle", "rb") as g:
            noise_analytics_ova.append(pickle.load(g))
    metrics_softmax.append(noise_analytics_softmax)
    metrics_ova.append(noise_analytics_ova)

with open(f'metrics/metrics_ova_expert_70_correct.json', "w") as f:
                json.dump(metrics_ova, f)

with open(f'metrics/metrics_softmax_expert4.json', "w") as f:
                json.dump(metrics_softmax, f)