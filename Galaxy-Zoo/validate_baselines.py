# To include lib
from __future__ import division

import sys

sys.path.insert(0, "../")

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict

import hemmer_baseline_trained
import main_classifier
import main_increase_experts_hard_coded
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from galaxyzoodataset import GalaxyZooDataset
from models.baseline import Resnet, Network
from models.experts import synth_expert
from models.resnet50 import ResNet50_defer
from scipy import stats
from torch.autograd import Variable
from tqdm import tqdm

from lib.losses import Criterion
from lib.utils import AverageMeter, accuracy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


expert1 = synth_expert(flip_prob=0.75, p_in=0.10)
expert2 = synth_expert(flip_prob=0.50, p_in=0.50)
expert3 = synth_expert(flip_prob=0.30, p_in=0.75)
expert4 = synth_expert(flip_prob=0.20, p_in=0.85)
available_experts = [expert1, expert2, expert3, expert4]
available_expert_fns = ["FlipHuman", "predict_prob", "predict_random"]

experts = [
    getattr(expert2, "predict_random"),
    getattr(expert1, "predict_prob"),
    getattr(expert2, "FlipHuman"),
    getattr(expert3, "predict_prob"),
    getattr(expert3, "FlipHuman"),
    getattr(expert4, "FlipHuman"),
    getattr(expert4, "predict_prob"),
    getattr(expert4, "HumanExpert"),
    getattr(expert2, "predict_prob"),
    getattr(expert1, "HumanExpert"),
]


def feed_result_dict_seed(result_dict_seed, result, i):
    experiment_expert_i = "experiment_expert_" + str(i)

    for key, val in result_dict_seed.items():
        if key == "coverage":  # coverage
            total_samples = int(result["test"]["coverage"].split()[-1].split("f")[-1])
            covered_samples = int(result["test"]["coverage"].split()[0])
            coverage = covered_samples / total_samples
            result_dict_seed[key].append(coverage)
        elif key == experiment_expert_i:
            for j in range(i):
                result_dict_seed[key]["expert_" + str(j)] = result["test"][
                    "expert_" + str(j)
                ]
        else:
            if "experiment" in key:
                continue  # already filled
            else:
                result_dict_seed[key].append(result["test"][key])
    return result_dict_seed


def feed_result_dict(result_dict, result_dict_seed):
    for key, val in result_dict.items():
        if "experiment" in key:
            for expert in result_dict[key].keys():
                result_dict[key][expert].append(result_dict_seed[key][expert])
        else:
            result_dict[key].append(result_dict_seed[key])
    return result_dict


def fill_result_dict_seed_experts_dict(result_dict_seed, experiment_experts):
    experiment_expert_i = "experiment_expert_" + str(experiment_experts)
    result_dict_seed[experiment_expert_i] = {
        "expert_" + str(i): None for i in range(experiment_experts)
    }
    return result_dict_seed


def print_results(result_dict):
    for key, val in result_dict.items():
        if "experiment" not in key:
            print("=== {} Mean and Standard Error===".format(key))
            print("Mean {}".format(np.mean(np.array(val), axis=0)))
            print("Standard Error {}".format(stats.sem(np.array(val), axis=0)))
        else:
            print("=== {} Experiment ===".format(key.split("_")[-1]))
            for expert, v in result_dict[key].items():
                print("{} Mean {}".format(expert, np.mean(np.array(v), axis=0)))
                print(
                    "{} Standard Error {}".format(
                        expert, stats.sem(np.array(v), axis=0)
                    )
                )
                print("==============")

    return


def main_validate_best_expert(testD, expert_fns, config):
    kwargs = {"num_workers": 1, "pin_memory": True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs
    )
    expert_accs = {k: 0 for k in range(len(expert_fns))}
    for i, fn in enumerate(expert_fns):
        exp_acc = 0.0
        for j, (inp, lbl, Z) in enumerate(test_dl, 1):
            exp_pred = fn(inp, lbl, Z)
            exp_pred = torch.tensor(exp_pred)
            exp_acc += torch.mean(lbl.eq(exp_pred).float())
        expert_accs[i] = exp_acc / j
    return expert_accs


def validate_best_expert(config):
    testD = GalaxyZooDataset(split="test")
    experiment_experts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracy = []
    for seed in ["", 948, 625, 436, 791]:
        if seed != "":
            set_seed(seed)
        acc = []
        expert_fns = []
        for i, n in tqdm(enumerate(experiment_experts)):
            expert_fns.append(experts[i])
            expert_accs = main_validate_best_expert(testD, expert_fns, config)
            temp = expert_accs.values()
            best = max(temp)
            # print(expert_accs)
            acc.append(best)
        # print(acc)
        accuracy.append(acc)

    print("===Mean and Standard Error===")
    print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def filter(dict_):
    d = {}
    for k, v in dict_.items():
        if torch.is_tensor(v):
            v = v.item()
        d[k] = v
    return d


def forward_surrogate(model, dataloader, expert_fns, config):
    confidence = []
    true = []
    expert_predictions = defaultdict(list)

    with torch.no_grad():
        for inp, lbl, Z in dataloader:
            inp = inp.to(device)
            lbl = lbl.to(device)
            Z = Z
            conf = model(inp)
            for i, fn in enumerate(expert_fns):
                expert_pred1 = fn(inp, lbl, Z)
                expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl.cpu())
            # break

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(
        -1, config["n_classes"] + len(expert_fns)
    )
    for k, v in expert_predictions.items():
        expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(
            -1
        )

    return true, confidence, [v.numpy() for k, v in expert_predictions.items()]


def main_validate_surrogate(model, testD, expert_fns, config, seed=""):
    def get(severity, dl):
        true, confidence, expert_predictions = forward_surrogate(
            model, dl, expert_fns, config
        )

        print(
            "shapes: true labels {}, confidences {}, expert_predictions {}".format(
                true.shape, confidence.shape, np.array(expert_predictions).shape
            )
        )

        criterion = Criterion()
        loss_fn = getattr(criterion, config["loss_type"])
        print("Evaluate...")
        result_ = main_increase_experts_hard_coded.evaluate(
            model,
            expert_fns,
            loss_fn,
            config["n_classes"] + len(expert_fns),
            dl,
            config,
        )
        # print(result_)
        result[severity] = filter(result_)
        true_label[severity] = true.numpy()
        classifier_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions

    model_path = os.path.join(
        config["ckp_dir"],
        config["experiment_name"]
        + "_"
        + str(len(expert_fns))
        + "_experts"
        + "_seed_"
        + str(seed),
    )
    model.load_state_dict(torch.load(model_path + ".pt", map_location=device))
    model = model.to(device)

    model_name = (
        config["experiment_name"]
        + "_"
        + str(len(expert_fns))
        + "_experts"
        + "_seed_"
        + str(seed)
    )

    kwargs = {"num_workers": 1, "pin_memory": True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs
    )

    result = {}
    classifier_confidence = {}
    true_label = {}
    expert_preds = {}

    get("test", test_dl)

    with open(config["ckp_dir"] + "/true_label_" + model_name + ".txt", "w") as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + "/confidence_" + model_name + ".txt", "w") as f:
        json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

    with open(
        config["ckp_dir"] + "/expert_predictions_" + model_name + ".txt", "w"
    ) as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    with open(
        config["ckp_dir"] + "/validation_results_" + model_name + ".txt", "w"
    ) as f:
        json.dump(json.dumps(result, cls=NumpyEncoder), f)

    return result


def validate_surrogate(config):
    config["ckp_dir"] = (
        "./" + config["loss_type"] + "_increase_experts_select_hard_coded"
    )
    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8, 9]
    # experiment_experts = [2]

    # Result dict ===
    result_dict = {"system_accuracy": [], "expert_accuracy": [], "coverage": []}
    result_dict = {
        **result_dict,
        **{
            "experiment_expert_" + str(i): {"expert_" + str(j): [] for j in range(i)}
            for i in experiment_experts
        },
    }

    accuracy = []
    for seed in ["", 948, 625]:
        # for seed in ['']:

        if seed != "":
            set_seed(seed)

        result_dict_seed = {k: [] for k in result_dict.keys()}
        acc = []
        for i, n in tqdm(enumerate(experiment_experts)):

            result_dict_seed = fill_result_dict_seed_experts_dict(result_dict_seed, n)
            print("n is {}".format(n))
            num_experts = n

            expert_fns = [experts[j] for j in range(n)]

            model = model = ResNet50_defer(int(config["n_classes"]) + num_experts)

            testD = GalaxyZooDataset(split="test")
            result = main_validate_surrogate(
                model, testD, expert_fns, config, seed=seed
            )

            result_dict_seed = feed_result_dict_seed(result_dict_seed, result, n)
            # acc.append(result['test']['system_accuracy'])

        # for key, val in result_dict.items():
        #         result_dict[key].append(result_dict_seed[key])
        result_dict = feed_result_dict(result_dict, result_dict_seed)
        # accuracy.append(acc)

    print("==={}===".format(config["loss_type"]))
    print_results(result_dict)
    with open(
        config["ckp_dir"] + "/results_" + config["ckp_dir"].split("/")[-1] + ".txt", "w"
    ) as f:
        json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

    # print("=== Sys. Acc. Mean and Standard Error===")
    # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def forward_hemmer(model, dataloader, expert_fns):
    confidence = []
    true = []
    expert_predictions = defaultdict(list)

    with torch.no_grad():
        for inp, lbl, Z in dataloader:
            inp = inp.to(device)
            lbl = lbl.to(device)
            Z = Z
            feature_extractor = model[0]
            allocator = model[1]  # experts confs
            bath_features = feature_extractor(inp)
            conf = allocator(bath_features)
            for i, fn in enumerate(expert_fns):
                expert_pred1 = fn(inp, lbl, Z)
                expert_predictions[i].append(expert_pred1)
            confidence.append(conf.cpu())
            true.append(lbl.cpu())
            # break

    true = torch.stack(true, dim=0).view(-1)
    confidence = torch.stack(confidence, dim=0).view(-1, len(expert_fns) + 1)
    for k, v in expert_predictions.items():
        expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(
            -1
        )

    return true, confidence, [v.numpy() for k, v in expert_predictions.items()]


def main_validate_hemmer(model, testD, expert_fns, config, seed=""):
    def get(severity, dl):
        true, confidence, expert_predictions = forward_hemmer(model, dl, expert_fns)

        print(
            "shapes: true labels {}, confidences {}, expert_predictions {}".format(
                true.shape, confidence.shape, np.array(expert_predictions).shape
            )
        )

        criterion = Criterion()
        loss_fn = nn.NLLLoss()
        print("Evaluate...")
        result_ = hemmer_baseline_trained.evaluate(
            model,
            expert_fns,
            loss_fn,
            config["n_classes"] + len(expert_fns),
            dl,
            config,
        )
        # print(result_)
        result[severity] = filter(result_)
        true_label[severity] = true.numpy()
        allocator_confidence[severity] = confidence.numpy()
        expert_preds[severity] = expert_predictions

    model_path = os.path.join(
        config["ckp_dir"],
        config["experiment_name"]
        + "_"
        + str(len(expert_fns))
        + "_experts"
        + "_seed_"
        + str(seed),
    )
    load_dict = torch.load(model_path + ".pt", map_location=device)
    feature_extractor, allocator, classifier = model[0], model[1], model[2]

    # print(type(load_dict['allocator_state_dict']),
    #       type(load_dict['classifier_state_dict']()))
    allocator.load_state_dict(load_dict["allocator_state_dict"])
    # import copy  # Careful with this. Actually I saved the method instead of the state_dict() for classifier
    # classifier.load_state_dict(copy.deepcopy(
    #     load_dict['classifier_state_dict']()))
    # feature_extractor.load_state_dict(copy.deepcopy(
    #     load_dict['feature_extractor_state_dict']()))

    classifier.load_state_dict(load_dict["classifier_state_dict"])
    feature_extractor.load_state_dict(load_dict["feature_extractor_state_dict"])
    feature_extractor, allocator, classifier = (
        feature_extractor.to(device),
        allocator.to(device),
        classifier.to(device),
    )
    model = (feature_extractor, allocator, classifier)

    model_name = (
        config["experiment_name"]
        + "_"
        + str(len(expert_fns))
        + "_experts"
        + "_seed_"
        + str(seed)
    )
    kwargs = {"num_workers": 1, "pin_memory": True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs
    )

    result = {}
    allocator_confidence = {}
    true_label = {}
    expert_preds = {}

    get("test", test_dl)

    with open(config["ckp_dir"] + "/true_label_" + model_name + ".txt", "w") as f:
        json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

    with open(config["ckp_dir"] + "/confidence_" + model_name + ".txt", "w") as f:
        json.dump(json.dumps(allocator_confidence, cls=NumpyEncoder), f)

    with open(
        config["ckp_dir"] + "/expert_predictions_" + model_name + ".txt", "w"
    ) as f:
        json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

    with open(
        config["ckp_dir"] + "/validation_results_" + model_name + ".txt", "w"
    ) as f:
        json.dump(json.dumps(result, cls=NumpyEncoder), f)

    return result


def validate_hemmer(config):
    config["loss_type"] = "hemmer"
    config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts"
    config["experiment_name"] = "multiple_experts_hardcoded"
    # experiment_experts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8, 9]

    # Result dict ===
    result_dict = {"system_accuracy": [], "expert_accuracy": [], "coverage": []}
    result_dict = {
        **result_dict,
        **{
            "experiment_expert_" + str(i): {"expert_" + str(j): [] for j in range(i)}
            for i in experiment_experts
        },
    }

    accuracy = []
    # for seed in ['', 948,  625, 791, 436]:
    for seed in ["", 948, 625]:
        # for seed in ['']:

        if seed != "":
            set_seed(seed)
        expert_fns = []
        result_dict_seed = {k: [] for k in result_dict.keys()}
        # acc = []
        for i, n in tqdm(enumerate(experiment_experts)):

            result_dict_seed = fill_result_dict_seed_experts_dict(result_dict_seed, n)

            print("n is {}".format(n))
            num_experts = n
            # getattr(selected_expert, selected_expert_fn)
            expert_fns = [experts[j] for j in range(n)]

            # === Galaxy-Zoo models ===
            # print(len(expert_fns))
            feature_extractor = Resnet()
            classifier = Network(output_size=int(config["n_classes"]))
            allocator = Network(output_size=len(expert_fns) + 1)
            model = (feature_extractor, allocator, classifier)
            testD = GalaxyZooDataset(split="val")
            result = main_validate_hemmer(model, testD, expert_fns, config, seed=seed)

            result_dict_seed = feed_result_dict_seed(result_dict_seed, result, n)
            # acc.append(result['test']['system_accuracy'])

        # for key, val in result_dict.items():
        #         result_dict[key].append(result_dict_seed[key])
        result_dict = feed_result_dict(result_dict, result_dict_seed)
        # accuracy.append(acc)

    print("===HEMMER===")
    print_results(result_dict)
    with open(
        config["ckp_dir"] + "/results_" + config["ckp_dir"].split("/")[-1] + ".txt", "w"
    ) as f:
        json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

    # print("=== Sys. Acc. Mean and Standard Error===")
    # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def validate_hemmer_trained(config):
    config["loss_type"] = "hemmer"
    config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts_trained"
    config["experiment_name"] = "multiple_experts_hardcoded"
    experiment_experts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    experiment_experts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # experiment_experts = [8]

    # Result dict ===
    result_dict = {"system_accuracy": [], "expert_accuracy": [], "coverage": []}
    result_dict = {
        **result_dict,
        **{
            "experiment_expert_" + str(i): {"expert_" + str(j): [] for j in range(i)}
            for i in experiment_experts
        },
    }

    accuracy = []
    # for seed in ['', 948,  625, 791, 436]:
    for seed in ["", 948, 625]:
        # for seed in ['']:

        if seed != "":
            set_seed(seed)
        expert_fns = []
        result_dict_seed = {k: [] for k in result_dict.keys()}

        # acc = []
        for i, n in tqdm(enumerate(experiment_experts)):

            result_dict_seed = fill_result_dict_seed_experts_dict(result_dict_seed, n)

            print("n is {}".format(n))
            num_experts = n
            # getattr(selected_expert, selected_expert_fn)
            expert_fns = [experts[j] for j in range(n)]

            # === Galaxy-Zoo models ===
            # print(len(expert_fns))
            feature_extractor = Resnet()
            classifier = Network(output_size=int(config["n_classes"]))
            allocator = Network(output_size=len(expert_fns) + 1)
            model = (feature_extractor, allocator, classifier)
            testD = GalaxyZooDataset(split="test")
            result = main_validate_hemmer(model, testD, expert_fns, config, seed=seed)

            result_dict_seed = feed_result_dict_seed(result_dict_seed, result, n)
            # acc.append(result['test']['system_accuracy'])

        # for key, val in result_dict.items():
        #         result_dict[key].append(result_dict_seed[key])
        result_dict = feed_result_dict(result_dict, result_dict_seed)
        # accuracy.append(acc)

    print("===HEMMER TRAINED===")
    print_results(result_dict)
    with open(
        config["ckp_dir"] + "/results_" + config["ckp_dir"].split("/")[-1] + ".txt", "w"
    ) as f:
        json.dump(json.dumps(result_dict, cls=NumpyEncoder), f)

    # print("=== Sys. Acc. Mean and Standard Error===")
    # print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
    # print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))


def main_validate_classifier(model, testD, expert_fns, config, seed=""):
    model_name = (
        config["experiment_name"]
        + "_"
        + str(len(expert_fns))
        + "_experts"
        + "_seed_"
        + str(seed)
    )
    model_path = config["ckp_dir"] + "/" + model_name + ".pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    kwargs = {"num_workers": 1, "pin_memory": True}
    test_dl = torch.utils.data.DataLoader(
        testD, batch_size=config["batch_size"], shuffle=False, drop_last=True, **kwargs
    )

    result = {}
    result_ = main_classifier.evaluate(
        model, expert_fns, nn.NLLLoss(), config["n_classes"], test_dl, config
    )
    result["test"] = filter(result_)
    return result


def validate_classifier(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_classifier"
    expert_fns = []
    accuracy = []
    # , 948,  625,  436,  791]: #, 1750,  812, 1331, 1617,  650, 1816]:
    for seed in tqdm(["", 948, 625, 436, 791]):
        # for seed in ['']:
        print("run for seed {}".format(seed))
        if seed != "":
            set_seed(seed)
        model = model = ResNet50_defer(int(config["n_classes"]))
        testD = GalaxyZooDataset(split="test")
        result = main_validate_classifier(model, testD, expert_fns, config, seed=seed)
        accuracy.append(result["test"]["system_accuracy"])

    print("===Mean and Standard Error===")
    print("Mean {}".format(np.mean(np.array(accuracy))))
    print("Standard Error {}".format(stats.sem(np.array(accuracy))))


if __name__ == "__main__":
    # config surrogate loss methods

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="scaling parameter for the loss function, default=1.0.",
    )
    parser.add_argument(
        "--expert_type",
        type=str,
        default="predict_prob",
        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.",
    )
    parser.add_argument(
        "--n_classes", type=int, default=2, help="K for K class classification."
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ova",
        help="surrogate loss type for learning to defer.",
    )
    parser.add_argument(
        "--ckp_dir",
        type=str,
        default="./Models",
        help="directory name to save the checkpoints.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="multiple_experts",
        help="specify the experiment name. Checkpoints will be saved with this name.",
    )

    config = parser.parse_args().__dict__

    config["loss_type"] = "softmax"

    print("validate softmax surrogate loss method...")
    validate_surrogate(config)

    config["loss_type"] = "ova"

    print("validate ova surrogate loss method...")
    validate_surrogate(config)

    config["loss_type"] = "hemmer"

    print("validate Hemmer MoE baseline method...")
    validate_hemmer(config)

    config["loss_type"] = "hemmer"

    print("validate Hemmer TRAINED MoE baseline method...")
    validate_hemmer_trained(config)

    print("validate one classifier baseline...")
    config["loss_type"] = "softmax"
    config["experiment_name"] = "classifier"
    validate_classifier(config)

    print("validate best expert baseline...")
    validate_best_expert(config)
