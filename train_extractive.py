#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.MoE import build_exercise_condition_map
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
from torch.utils.data.dataloader import DataLoader
from Embedder.data_loader import Video_Loader
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def _requested_moe_architecture(args):
    architecture = {
        'moe': args.moe,
        'expert_type': args.expert_type,
        'expert_mode': args.expert_mode,
        'top_k': args.top_k,
        'expert_dim': args.expert_dim,
    }
    if args.expert_mode == 'tcn':
        architecture.update({
            'tcn_conv_type': args.tcn_conv_type,
            'tcn_channels': args.tcn_channels,
        })
    return architecture


def _checkpoint_moe_architecture(saved_moe_config):
    """Normalize current and legacy checkpoint MoE architecture fields."""
    saved_moe_config = dict(saved_moe_config)
    if 'moe' in saved_moe_config:
        architecture = {
            'moe': saved_moe_config['moe'],
            'expert_type': saved_moe_config.get('expert_type', 'joint'),
            'expert_mode': saved_moe_config.get('expert_mode', 'baseline'),
            'top_k': saved_moe_config.get('top_k', 3),
            'expert_dim': saved_moe_config.get('expert_dim', 256),
        }
        if architecture['expert_mode'] == 'tcn':
            architecture.update({
                'tcn_conv_type': saved_moe_config.get(
                    'tcn_conv_type', 'conv1d'),
                'tcn_channels': saved_moe_config.get('tcn_channels', 128),
            })
        return architecture

    legacy_type = saved_moe_config.get('condition_moe_type', 'none')
    if legacy_type in ('joint_condition', 'joint_group_condition'):
        raise ValueError(
            "checkpoint uses a removed condition-specific expert mode: {}"
            .format(legacy_type))
    legacy_expert_types = {
        'none': 'joint',
        'joint': 'joint',
        'joint_group': 'group',
        'exercise': 'exercise',
    }
    if legacy_type not in legacy_expert_types:
        raise ValueError(
            "unsupported legacy condition_moe_type: {}".format(legacy_type))
    legacy_top_k_keys = {
        'joint': ('joint_top_k', 5),
        'joint_group': ('joint_group_top_k', 3),
        'exercise': ('exercise_top_k', 1),
    }
    if legacy_type == 'none':
        top_k = 3
    else:
        top_k_key, top_k_default = legacy_top_k_keys[legacy_type]
        top_k = saved_moe_config.get(top_k_key, top_k_default)
    return {
        'moe': legacy_type != 'none',
        'expert_type': legacy_expert_types[legacy_type],
        'expert_mode': 'baseline',
        'top_k': top_k,
        'expert_dim': 256,
    }


def _validate_checkpoint_moe_config(args, checkpoint):
    saved_moe_config = checkpoint.get('moe_config')
    if saved_moe_config is None:
        raise KeyError("condition MoE checkpoint has no moe_config")
    saved_architecture = _checkpoint_moe_architecture(saved_moe_config)
    requested_architecture = _requested_moe_architecture(args)
    mismatched_config = {
        key: (saved_architecture.get(key), value)
        for key, value in requested_architecture.items()
        if saved_architecture.get(key) != value
    }
    if mismatched_config:
        raise ValueError(
            "checkpoint MoE config differs from requested args: {}"
            .format(mismatched_config))


def train_multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_ext(args, config, device_id):
    validate(args, config, device_id)


def validate(args, config, device_id):
    device = (
        torch.device('cpu')
        if args.visible_gpus == '-1' or device_id < 0
        else torch.device('cuda:{}'.format(device_id))
    )

    # Variable-width exercise expert heads must be reconstructed before the
    # model state can be loaded.
    ckpt_path = args.bert_validate_ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    if args.moe:
        _validate_checkpoint_moe_config(args, ckpt)
        if args.expert_type == 'exercise':
            exercise_condition_map = ckpt.get('exercise_condition_map')
            if exercise_condition_map is None:
                raise KeyError(
                    "exercise MoE checkpoint has no exercise_condition_map")
            args.exercise_condition_map = exercise_condition_map

    model = ExtSummarizer(args, device, checkpoint=None)
    # here #
    model.eval()
    # from torch.utils.data.dataloader import DataLoader
    # from Embedder.data_loader import Video_Loader
    # from Embedder.Embedder_config import config
    # video_dataset = Video_Loader(config=config, data_path='/storage/hjchoi/BERTSUMFORHPE/embedder_valid.json')
    # video_loader = torch.utils.data.DataLoader(
    #     video_dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     collate_fn=video_dataset.collate_fn)
    from torch.utils.data.dataloader import DataLoader
    from Embedder.Embedder_API import Embedder
    video_dataset = Video_Loader(config=config, mode=args.mode)

    video_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn)

    trainer = build_trainer(
        args, config, device_id, model, None,
        video_dataset=video_dataset)

    # Load Multi-Head Model from Checkpoint
    new_state_dict = {}

    checkpoint_model = ckpt.get("model_state_dict", ckpt.get("model"))
    if checkpoint_model is None:
        raise KeyError(
            "checkpoint does not contain model_state_dict or model")

    for key, value in checkpoint_model.items():
        if key.startswith("bert.encoder."):
            new_key = key.replace("bert.encoder.", "bert.model.encoder.", 1)
        elif key.startswith("bert.embeddings."):
            new_key = key.replace("bert.embeddings.", "bert.model.embeddings.", 1)
        elif key.startswith("bert.pooler."):
            new_key = key.replace("bert.pooler.", "bert.model.pooler.", 1)
        else:
            new_key = key

        new_state_dict[new_key] = value

    current_state_dict = trainer.model.state_dict()
    mismatched = [key for key, value in new_state_dict.items() if key in current_state_dict and current_state_dict[key].shape != value.shape]
    if mismatched:
        raise RuntimeError(
            "checkpoint architecture mismatch: {}".format(mismatched))

    missing, unexpected = trainer.model.load_state_dict(new_state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError("checkpoint state mismatch: missing={}, unexpected={}".format(missing, unexpected))

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    trainer.embedder.load_state_dict(ckpt["embedder_state_dict"], strict=True)

    trainer.model.eval()
    trainer.embedder.eval()
    #

    stats = trainer.validate(video_loader, video_dataset)
    return stats.xent()


def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    new_state_dict = {}
    checkpoint_model = checkpoint.get(
        'model', checkpoint.get('model_state_dict'))
    if checkpoint_model is None:
        raise KeyError(
            "checkpoint does not contain model or model_state_dict")
    for key, value in checkpoint_model.items():
        if key.startswith('bert.encoder.'):
            continue
        elif key.startswith('bert.model.'):
            new_state_dict[key] = value
        elif key.startswith('ext_layer.'):
            new_state_dict[key] = value
        elif key.startswith('embeddings.') or key.startswith('encoder.') or key.startswith('pooler.'):
            new_key = 'bert.model.' + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    modified_checkpoint = checkpoint.copy()
    modified_checkpoint['model'] = new_state_dict
    if 'opt' in checkpoint:
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, modified_checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)

def train_ext(args, config, device_id):
    if (args.world_size > 1):
        train_multi_ext(args)
    else:
        train_single_ext(args, config, args.device_id)


def train_single_ext(args, config, device_id):
    if args.log_file is not None:
        init_logger(args.log_file)

    if args.device_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.device_id))
        torch.cuda.set_device(args.device_id)

    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)
    #
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    preloaded_train_dataset = None
    if args.moe and args.expert_type == 'exercise':
        preloaded_train_dataset = Video_Loader(
            config=config, mode=args.mode)
        args.exercise_condition_map = build_exercise_condition_map(
            preloaded_train_dataset.videos,
            num_exercises=config.CLASS_NUM,
            num_conditions=config.NUM_CONDITIONS,
        )

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        if args.moe:
            _validate_checkpoint_moe_config(args, checkpoint)
            if args.expert_type == 'exercise':
                saved_mapping = checkpoint.get('exercise_condition_map')
                if saved_mapping is None:
                    raise KeyError(
                        "exercise MoE checkpoint has no "
                        "exercise_condition_map")
                current_mapping = getattr(
                    args, 'exercise_condition_map', saved_mapping)
                if current_mapping != saved_mapping:
                    raise ValueError(
                        "checkpoint exercise_condition_map differs from "
                        "TRAIN data")
                args.exercise_condition_map = saved_mapping
        new_state_dict = {}
        checkpoint_model = checkpoint.get(
            'model', checkpoint.get('model_state_dict'))
        if checkpoint_model is None:
            raise KeyError(
                "checkpoint does not contain model or model_state_dict")
        for key, value in checkpoint_model.items():
            if key.startswith('bert.encoder.'):
                continue
            elif key.startswith('bert.model.'):
                new_state_dict[key] = value
            elif key.startswith('ext_layer.'):
                new_state_dict[key] = value
            elif key.startswith('embeddings.') or key.startswith('encoder.') or key.startswith('pooler.'):
                new_key = 'bert.model.' + key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        modified_checkpoint = checkpoint.copy()
        modified_checkpoint['model'] = new_state_dict

        if args.bert_random_init:
            sample_keys = [k for k in modified_checkpoint['model'].keys()
                           if k.startswith('bert.model.encoder.layer.0.attention.self.query')]
            before_stats = {}
            for key in sample_keys:
                before_stats[key] = {
                    'mean': modified_checkpoint['model'][key].mean().item(),
                    'std': modified_checkpoint['model'][key].std().item(),
                    'sum': modified_checkpoint['model'][key].sum().item()
                }

            init_count = 0
            for key,value in modified_checkpoint['model'].items():
                if key.startswith('bert.model.encoder.'):
                    if 'weight' in key:
                        if 'LayerNorm' in key:
                            torch.nn.init.ones_(modified_checkpoint['model'][key])
                        else:
                            torch.nn.init.xavier_uniform_(modified_checkpoint['model'][key])
                        init_count += 1
                    elif 'bias' in key:
                        torch.nn.init.zeros_(modified_checkpoint['model'][key])
                        init_count += 1
            for key in sample_keys:
                after_mean = modified_checkpoint['model'][key].mean().item()
                after_std = modified_checkpoint['model'][key].std().item()
                after_sum = modified_checkpoint['model'][key].sum().item()

                print(f'Key: {key}')
                print(
                    f'  Before - mean: {before_stats[key]["mean"]:.6f}, std: {before_stats[key]["std"]:.6f}, sum: {before_stats[key]["sum"]:.4f}')
                print(f'  After  - mean: {after_mean:.6f}, std: {after_std:.6f}, sum: {after_sum:.4f}')

                changed = abs(before_stats[key]["sum"] - after_sum) > 0.0001
                print(f'  Changed: {changed}')

            print(f'Total reinitialized BERT parameters: {init_count}')
        if args.embedder_random_init:
            sample_keys = [k for k in modified_checkpoint['model'].keys()
                           if k.startswith('bert.model.embeddings.word_embeddings.weight')]

            # Store statistics before initialization
            before_stats = {}
            for key in sample_keys:
                before_stats[key] = {
                    'mean': modified_checkpoint['model'][key].mean().item(),
                    'std': modified_checkpoint['model'][key].std().item(),
                    'sum': modified_checkpoint['model'][key].sum().item()
                }

            init_count = 0
            for key, value in modified_checkpoint['model'].items():
                # Initialize embeddings layers (word, position, token_type embeddings)
                if key.startswith('bert.model.embeddings.'):
                    if 'weight' in key:
                        if 'LayerNorm' in key:
                            torch.nn.init.ones_(modified_checkpoint['model'][key])
                        else:
                            torch.nn.init.xavier_uniform_(modified_checkpoint['model'][key])
                        init_count += 1
                    elif 'bias' in key:
                        torch.nn.init.zeros_(modified_checkpoint['model'][key])
                        init_count += 1

            # Print verification statistics
            for key in sample_keys:
                after_mean = modified_checkpoint['model'][key].mean().item()
                after_std = modified_checkpoint['model'][key].std().item()
                after_sum = modified_checkpoint['model'][key].sum().item()
                print(f'Key: {key}')
                print(
                    f'  Before - mean: {before_stats[key]["mean"]:.6f}, std: {before_stats[key]["std"]:.6f}, sum: {before_stats[key]["sum"]:.4f}')
                print(f'  After  - mean: {after_mean:.6f}, std: {after_std:.6f}, sum: {after_sum:.4f}')
                changed = abs(before_stats[key]["sum"] - after_sum) > 0.0001
                print(f'  Changed: {changed}')
            print(f'Total reinitialized embedder parameters: {init_count}')
        if 'opt' in checkpoint:
            opt = vars(checkpoint['opt'])
            for k in opt.keys():
                if k in model_flags:
                    setattr(args, k, opt[k])
    else:
        modified_checkpoint = None
        checkpoint = None

    model = ExtSummarizer(args, device, modified_checkpoint)
    optim = model_builder.build_optim(args, model, modified_checkpoint) # BERTModel weight

    logger.info(model)

    trainer = build_trainer(
        args, config, device_id, model, optim,
        video_dataset=preloaded_train_dataset)
    trainer.train(device, args.train_steps)
