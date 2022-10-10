import numpy as np
import json

METHOD_METRIC = ['deferral', 'idx_cal', 'coverage_cal', 'coverage_test', 'qhat']


# Json functions ===
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_dict_as_txt(dict, path):
    with open(path, 'w') as f:
        json.dump(json.dumps(dict, cls=NumpyEncoder), f)


def load_dict_txt(path):
    with open(path, 'r') as f:
        dict = json.loads(json.load(f))
    return dict


def get_metric(results, seeds_list, exp_list, metric, method):
    r"""
    Obtain the desired metric from the results dict.
    Args:
        results: Dictionary with seed->exp->method->metric
        seeds_list: List with the available seeds
        exp_list: List with the experiment values, e.g. p_out = [0.1, 0.2,...]
        metric: Desired metric:
            - system_accuracy
            - expert_accuracy
            - classifier_accuracy
            - alone_classifier
        method:

    Returns:

    """
    metric_np = np.zeros((len(seeds_list), len(exp_list)))
    for i, seed in enumerate(seeds_list):
        if metric in METHOD_METRIC:
            exp_metric = np.array([results[seed][exp][metric] for exp in exp_list])
        else:  # implement methods: 'standard', 'last', 'random', 'voting', 'ensemble'
            exp_metric = np.array([results[seed][exp][method][metric] for exp in exp_list])
        metric_np[i, :] = exp_metric
    return metric_np

# def get_accuracies(results):
#     pass
#
#
# def get_coverage():
#     pass
#
#
# def get_avg_set_sizes():
#     pass