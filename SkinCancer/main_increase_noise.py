# To include lib
import sys
sys.path.insert(0, "../")
import json
import os
import random
import time
import numpy as np
import pickle5 as pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from SkinCancerdataset import SkinCancerDataset
from models.experts import synth_expert
from models.resnet50 import ResNet50_defer
from lib.losses import Criterion
from lib.utils import AverageMeter, accuracy
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model, expert_fns, loss_fn, n_classes, data_loader, config):
    """
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    """
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert Accuracies === #
    alpha = config["alpha"]
    losses = []
    with torch.no_grad():
        for data in data_loader:
            images, labels, hpred = data
            images, labels, hpred = images.to(device), labels.to(device), hpred
            outputs = model(images)
            if config["loss"] == "softmax":
                outputs = F.softmax(outputs, dim=1)
            if config["loss"] == "ova":
                outputs = F.sigmoid(outputs)

            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_prediction1 = fn(images, labels, hpred)
                m = [0] * batch_size
                m2 = [0] * batch_size
                for j in range(0, batch_size):
                    if exp_prediction1[j] == labels[j].item():
                        m[j] = 1
                        m2[j] = alpha
                    else:
                        m[j] = 0
                        m2[j] = 1

                m = torch.tensor(m)
                m2 = torch.tensor(m2)
                m = m.to(device)
                m2 = m2.to(device)
                collection_Ms.append((m, m2))
                expert_predictions.append(exp_prediction1)

            loss = loss_fn(outputs, labels, collection_Ms, n_classes)
            losses.append(loss.item())

            for i in range(0, batch_size):
                r = predicted[i].item() >= n_classes - len(expert_fns)
                prediction = predicted[i]
                if predicted[i] >= n_classes - len(expert_fns):
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes - len(expert_fns)):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    deferred_exp = (predicted[i] - (n_classes - len(expert_fns))).item()
                    # cdeferred_exp = ((n_classes - 1) - predicted[i]).item()  # reverse order, as in loss function
                    exp_prediction = expert_predictions[deferred_exp][i]
                    #
                    # Deferral accuracy: No matter expert ===
                    exp += exp_prediction == labels[i].item()
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[deferred_exp] += (
                        exp_prediction == labels[i].item()
                    )
                    expert_total_dic[deferred_exp] += 1
                    #
                    correct_sys += exp_prediction == labels[i].item()
                real_total += 1
    cov = total / real_total * 100

    #  === Individual Expert Accuracies === #
    expert_accuracies = {
        "expert_{}".format(str(k)): 100
        * expert_correct_dic[k]
        / (expert_total_dic[k] + 0.0002)
        for k in range(len(expert_fns))
    }
    # Add expert accuracies dict
    to_print = {
        "coverage": cov,
        "system_accuracy": 100 * correct_sys / real_total,
        "expert_accuracy": 100 * exp / (exp_total + 0.0002),
        "classifier_accuracy": 100 * correct / (total + 0.0001),
        "alone_classifier": 100 * alone_correct / real_total,
        "validation_loss": np.average(losses),
        "n_experts": len(expert_fns),
        **expert_accuracies,
    }
    print(to_print, flush=True)
    return to_print


def train_epoch(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch, expert_fns, loss_fn, n_classes, alpha,config):
    """ Train for one epoch """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, (input, target, hpred) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        target = target.to(device)
        input = input.to(device)
        hpred = hpred

        # compute output
        output = model(input)

        if config["loss"] == "softmax":
            output = F.softmax(output, dim=1)

        # get expert  predictions and costs
        batch_size = output.size()[0]  # batch_size
        collection_Ms = []
        # We only support \alpha=1
        for _, fn in enumerate(expert_fns):
            # We assume each expert function has access to the extra metadata, even if they don't use it.
            m = fn(input, target, hpred)
            m2 = [0] * batch_size
            for j in range(0, batch_size):
                if m[j] == target[j].item():
                    m[j] = 1
                    m2[j] = alpha
                else:
                    m[j] = 0
                    m2[j] = 1
            m = torch.tensor(m)
            m2 = torch.tensor(m2)
            m = m.to(device)
            m2 = m2.to(device)
            collection_Ms.append((m, m2))

        # compute loss
        loss = loss_fn(output, target, collection_Ms, n_classes)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not iters < warmup_iters:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters += 1

        if i % 10 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                ),
                flush=True,
            )

    return iters, np.average(epoch_train_loss)

def train(model, train_dataset, validation_dataset, expert_fns, config, seed=""):
    n_classes = config["n_classes"] + len(expert_fns)
    kwargs = {"num_workers": 0, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    model = model.to(device)
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), config["lr"], weight_decay=config["weight_decay"])
    criterion = Criterion()
    loss_fn = getattr(criterion, config["loss"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["epochs"])
    best_validation_loss = np.inf
    patience = 0
    iters = 0
    warmup_iters = config["warmup_epochs"] * len(train_loader)
    lrate = config["lr"]
    epoch_metrics = []

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch, expert_fns, loss_fn, n_classes, config["alpha"], config)
        metrics = evaluate(model, expert_fns, loss_fn, n_classes, valid_loader, config)
        metrics["train_loss"] = train_loss
        epoch_metrics.append(metrics)
        validation_loss = metrics["validation_loss"]

        if validation_loss < best_validation_loss:
            best_metric = metrics
            best_validation_loss = validation_loss
            print("Saving the model with classifier accuracy {}".format(metrics["classifier_accuracy"]), flush=True)
            save_path = os.path.join(
                config["ckp_dir"],
                config["experiment_name"]
                + "_"
                + config["loss"]
                + "_"
                + str(config["seed"])
                + "_"
                + str(config["noise_rate"]),
            )
            torch.save(model.state_dict(), save_path + ".pt")
            # Additionally save the whole config dict
            with open(save_path + ".json", "w") as f:
                json.dump(config, f)
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            print("Early Exiting Training.", flush=True)
            break

    return best_metric, epoch_metrics


expert = synth_expert(flip_prob=0.0, p_in=0.7)
expert_fn = [getattr(expert, "predict_prob")]


def increase_error(config):
    for seed in [42, 35, 936, 235, 464]:
        config["seed"] = seed
        for noise_rate in [0.0, 0.02, 0.04, 0.06, 0.08]:
            config["noise_rate"] = noise_rate
            for loss in ["softmax", "ova"]:
                config["loss"] = loss
                print(f"The current run is: {loss} {seed} {noise_rate}")
                model = ResNet50_defer(int(config["n_classes"]) + len(expert_fn))
                dataset_train = SkinCancerDataset(error_rate=noise_rate)
                dataset_validation = SkinCancerDataset(split="val", error_rate=noise_rate)
                metrics, epoch_metrics = train(model, dataset_train, dataset_validation, expert_fn, config, seed=seed)

                with open(f'metrics/{loss}/metrics_{loss}_{seed}_{noise_rate}.pickle', "wb") as f:
                    pickle.dump(metrics, f)
                with open(f'logs/{loss}/logs_{loss}_{seed}_{noise_rate}.json', "w") as f:
                    json.dump(epoch_metrics, f)


if __name__ == "__main__":
    config = dict()
    config["batch_size"] = 32
    config["alpha"] = 1.0
    config["epochs"] = 100
    config["patience"] = 25
    config["n_classes"] = 2
    config["lr"] = 0.001
    config["weight_decay"] = 5e-4
    config["warmup_epochs"] = 5
    config["ckp_dir"] = "models_checkpoints"
    config["experiment_name"] = "increase_error"
    increase_error(config)