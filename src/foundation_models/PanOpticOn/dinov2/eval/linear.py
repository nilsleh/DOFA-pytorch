# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.eval.wrapper import backbone_to_features
from dinov2.data import SamplerType, make_data_loader, make_dataset
import dinov2.distributed as distributed
from dinov2.eval.metrics import build_metric
from dinov2.eval.utils import  evaluate
from dinov2.logging import MetricLogger
from dinov2.utils.data import dict_to_device
from omegaconf import OmegaConf 
import time
import datetime
import math

logger = logging.getLogger("dinov2")


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m

def get_ds_name(ds_cfg):
    return ds_cfg.get('display_name', None) or ds_cfg['id']

def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, pooling, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.pooling = pooling
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list, **kwargs):
        output = backbone_to_features(x_tokens_list, self.use_n_blocks, self.pooling, **kwargs)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs, **kwargs):
        return {k: v.forward(inputs, **kwargs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None, **fwd_kwargs):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))
        self.fwd_kwargs = fwd_kwargs

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples, **self.fwd_kwargs)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(sample_output, pooling, n_last_blocks_list, learning_rates, batch_size, num_classes=1000, norm=None):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for pool in pooling:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = backbone_to_features(sample_output, use_n_blocks=n, pooling=pool, norm=norm).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, pooling=pool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_pooling_{pool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metric_type,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    class_mapping=None,
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)
    target_metric = list(metric.keys())[0]
    postprocessors = {k: LinearPostprocessor(v, class_mapping, norm=feature_model.norm) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    logger.info("")
    results_dict = {}
    max_val = 0
    best_classifier = ""
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric[target_metric].item() > max_val
        ) or classifier_string == best_classifier_on_val:
            max_val = metric[target_metric].item()
            best_classifier = classifier_string

    results_dict["best_classifier"] = {"name": best_classifier, 'max_val': max_val, 'metric_str': target_metric}

    logger.info(f"best classifier by {target_metric}: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


def eval_linear(
    *,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,

    max_iter,
    iter_per_epoch,
    eval_period_iter,
    checkpoint_period,  # In number of iter, creates a new file every period

    metric_type,
    training_num_classes,
    resume=True,
    classifier_fpath=None,
    val_class_mapping=None,
    criterion_cfg = {'id': 'CrossEntropyLoss'},
):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    checkpointer.logger = logger
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter, max_to_keep=1)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"
    criterion = make_criterion(criterion_cfg)

    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        iteration,
        epoch_len=iter_per_epoch,
    ):
        data = dict_to_device(data, torch.cuda.current_device(), non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features, norm=feature_model.norm)

        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if distributed.is_main_process():
            periodic_checkpointer.step(iteration)

        if eval_period_iter > 0 and (iteration + 1) % int(eval_period_iter) == 0 and iteration != max_iter - 1:
            torch.cuda.synchronize()
            _ = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,)
            torch.cuda.synchronize()


        iteration = iteration + 1


    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
        class_mapping=val_class_mapping,
    )
    return val_results_dict, feature_model, linear_classifiers, iteration


def make_eval_data_loader(config, batch_size, num_workers, pin_memory=True, persistent_workers=True):
    test_dataset = make_dataset(config)
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED if distributed.is_enabled() else SamplerType.EPOCH,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )
    return test_data_loader


def make_criterion(cfg):
    cfg = deepcopy(cfg)
    id = cfg.pop("id")
    logger.info(f"Criterion: {id} with cfg {cfg}")
    if id == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**cfg)
    elif id == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss(**cfg)
    else:
        raise ValueError(f"Unknown criterion {id}")


def test_on_datasets(
    feature_model,
    linear_classifiers,
    test_dataset_cfgs,
    batch_size,
    num_workers,
    test_metrics_list,
    metrics_file_path,
    training_num_classes,
    iteration,
    best_classifier_on_val,
    prefixstring="",
    test_class_mappings=[None],
):
    results_list = []
    for test_dataset_cfg, class_mapping, metric_type in zip(test_dataset_cfgs, test_class_mappings, test_metrics_list):
        ds_str = get_ds_name(test_dataset_cfg)
        logger.info(f"Testing on {ds_str}")
        test_data_loader = make_eval_data_loader(test_dataset_cfg, batch_size, num_workers, pin_memory=True)
        dataset_results_dict = evaluate_linear_classifiers(
            feature_model,
            remove_ddp_wrapper(linear_classifiers),
            test_data_loader,
            metric_type,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring="",
            class_mapping=class_mapping,
            best_classifier_on_val=best_classifier_on_val,
        )
        logger.info(f"Test Results for {ds_str}: {str(dataset_results_dict)}")

        dataset_results_dict = dataset_results_dict['best_classifier']
        dataset_results_dict['postfix'] = ds_str
        dataset_results_dict['val'] = round(dataset_results_dict.pop('max_val')*100, 2)
        results_list.append(dataset_results_dict)

    return results_list


def run_eval_linear(
    _model_wrapper_partial,
    output_dir,
    train_dataset_cfg,
    val_dataset_cfg,
    batch_size,
    num_workers,
    save_checkpoint_frequency_epoch,

    epochs,
    eval_period_epoch=None,
    eval_period_iter=None,
    iter_per_epoch=-1,

    heads=None,
    test_dataset_cfgs:List=[],
    resume=True,
    classifier_fpath=None,
    val_class_mapping_fpath=None,
    test_class_mapping_fpaths=[None],
    val_metrics=[{'id': 'MulticlassAccuracy'}],
    test_metrics_list=None,
    criterion_cfg = {'id': 'CrossEntropyLoss'},
):
    seed = 0

    test_metrics_list = [val_metrics] * len(test_dataset_cfgs)
    assert len(test_dataset_cfgs) == len(test_class_mapping_fpaths)

    train_dataset = make_dataset(train_dataset_cfg)
    handle = train_dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        handle = train_dataset.dataset
    training_num_classes = len(torch.unique(torch.Tensor(handle.get_targets().astype(int))))
    # sampler_type = SamplerType.SHARDED_INFINITE
    sampler_type = SamplerType.INFINITE

    # set max_iter & iter_per_epoch
    if iter_per_epoch == -1:
        total_nsamples = len(train_dataset) * epochs
        actual_bsz = batch_size * distributed.get_global_size()
        max_iter = math.ceil(total_nsamples / actual_bsz)
        iter_per_epoch = max_iter / epochs # float!
    else:
        max_iter = math.ceil(epochs * iter_per_epoch)

    # set eval_period_iter
    if eval_period_epoch is None and eval_period_iter is None:
        raise ValueError("must provide either eval_period_iter or eval_period_epoch")
    elif eval_period_epoch is not None and eval_period_iter is not None:
        raise ValueError("must provide exactly one of eval_period_iter or eval_period_epoch, got none")
    elif eval_period_epoch is not None:
        eval_period_iter = max(1.0, eval_period_epoch * iter_per_epoch)
    logger.info(f"max_iter: {max_iter}, iter_per_epoch: {iter_per_epoch}, eval_period_iter: {eval_period_iter}")
    checkpoint_period = math.ceil(save_checkpoint_frequency_epoch * iter_per_epoch)

    # classifiers

    n_last_blocks = max(heads.n_last_blocks_list)
    feature_model = _model_wrapper_partial(n_last_blocks=n_last_blocks)
    sample_dict = train_dataset[0][0]
    sample_dict['imgs'] = sample_dict['imgs'].unsqueeze(0).cuda()
    sample_dict['chn_ids'] = sample_dict['chn_ids'].unsqueeze(0).cuda()
    sample_output = feature_model(sample_dict)

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        heads.pooling,
        heads.n_last_blocks_list,
        heads.learning_rates,
        batch_size,
        training_num_classes,
        norm=feature_model.norm,
    )
    
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    checkpointer.logger = logger
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        # sampler_size= epoch_length * batch_size,
        sampler_advance=start_iter,
        persistent_workers=True,
        drop_last=False
    )
    val_data_loader = make_eval_data_loader(val_dataset_cfg, batch_size, num_workers, pin_memory=False, persistent_workers=True)


    if val_class_mapping_fpath is not None:
        logger.info(f"Using class mapping from {val_class_mapping_fpath}")
        val_class_mapping = np.load(val_class_mapping_fpath)
    else:
        val_class_mapping = None

    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter = max_iter,
        eval_period_iter=eval_period_iter,
        iter_per_epoch = iter_per_epoch,
        checkpoint_period=checkpoint_period,
        metric_type=val_metrics,
        training_num_classes=training_num_classes,
        resume=resume,
        val_class_mapping=val_class_mapping,
        classifier_fpath=classifier_fpath,
        criterion_cfg = criterion_cfg,
    )
    results_list = []
    if len(test_dataset_cfgs) > 0:
        results_list = test_on_datasets(
            feature_model,
            linear_classifiers,
            test_dataset_cfgs,
            batch_size,
            num_workers,  # 0
            test_metrics_list,
            metrics_file_path,
            training_num_classes,
            iteration,
            val_results_dict["best_classifier"]["name"],
            prefixstring="",
            test_class_mappings=test_class_mappings,
        )
    return results_list
