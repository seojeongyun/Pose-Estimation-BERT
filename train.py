#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import torch
import random
import numpy as np

from others.run_logging import start_run_logging


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SLACK_SERVER_NAME = '서버1번'
RUN_LOGGING = None

from Embedder.Embedder_config import config
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='baseline', type=str, choices=['bert', 'baseline'])
    # parser.add_argument("-mode", default='train-valid', type=str, choices=['train', 'train-valid', 'validate', 'test'])
    parser.add_argument("-mode", default='train-valid', type=str, choices=['train', 'train-valid', 'validate', 'test'])
    parser.add_argument("-bert_validate_ckpt",
                        default='/home/jysuh/PycharmProjects/BERTSUMFORHPE_multi_head_ver/multi_cls_model_save/best_model_ckpt.pt')
    parser.add_argument("-emb_mode", default=config.EMB_MODE, choices=['RELATIVE_BASIS', 'RELATIVE', 'BASIS'])
    # parser.add_argument("-bert_data_path", default='./bert_data/')
    parser.add_argument("-model_path", default='./model_save/')
    parser.add_argument("-multi_cls_model_path", default='./multi_cls_model_save')
    parser.add_argument("-result_path", default='./results/news')
    parser.add_argument("-temp_dir", default='./temp')

    # parser.add_argument("-batch_size", default=10, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=5e-6, type=float)
    parser.add_argument("-lr_dec", default=2e-6, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=4, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.3, type=float)
    parser.add_argument("-ext_layers", default=4, type=int) ###change NUM ENC_layer
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=5e-5, type=float) # 5e-3 better
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # Scheduler
    parser.add_argument("-use_scheduler", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-base_lr", default=5e-5)
    parser.add_argument("-max_lr", default=5e-3) # 5e-4
    parser.add_argument("-step_size_up", default=20, type=int)
    parser.add_argument("-cycle_momentum", default=False)

    parser.add_argument("-save_checkpoint_steps", default=50, type=int)
    parser.add_argument("-save_checkpoint_epoch", default=5, type=int)
    parser.add_argument("-accum_count", default=10, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_epoch",default=50, type=int)
    parser.add_argument("-train_steps", default=100, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-device_id", default='1', type=int)
    parser.add_argument('-visible_gpus', default='1', type=str)
    parser.add_argument('-gpu_ranks', default='1', type=str)
    parser.add_argument('-log_file', default=None)
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='') # ./model_save/RELATIVE_BASIS/basis_False/relative_FalseNUM_EMB_LAYER:4/model_ENCLAYER_2.pt
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    ###########################################
    parser.add_argument("-bert_data_path", default='/storage/hjchoi/BERTSUMFORHPE/bert_data')
    parser.add_argument("-batch_size", default=config.BATCH_SIZE, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    #
    parser.add_argument("-pad_id", default=0, type=int)
    parser.add_argument("-sep_id", default=1, type=int)
    parser.add_argument("-weight_path", default='./model_save/embedding_weights/[new]nn_embedding_model.pt')
    ##############################

    parser.add_argument("-bert_random_init", default=False)
    parser.add_argument("-embedder_random_init", default=False)

    parser.add_argument("-cond_loss_weight", default=1.0,type=float)
    parser.add_argument("-ex_loss_weight", default=1.0, type=float)

    parser.add_argument("-weighted_loss", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-weighted_loss_value", default=1.0, type=float)
    parser.add_argument("-threshold", default=0.65, type=float)
    parser.add_argument("-attach_cls_token_to_end_of_seqlen", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-condition_pr_analysis", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-condition_pr_output", default=None)
    parser.add_argument("-exercise_ap_roc_analysis", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-exercise_ap_roc_output", default=None)

    # Condition Mixture-of-Experts
    parser.add_argument("-moe", type=str2bool, nargs='?', default=True)
    parser.add_argument("-expert_type", default='joint', choices=['joint', 'group', 'exercise'])
    parser.add_argument("-expert_mode", default='tcn', choices=['baseline', 'tcn'])
    parser.add_argument("-tcn_conv_type", default='depth_point_wise', choices=['conv1d', 'depth_point_wise'])
    parser.add_argument("-tcn_channels", default=128, type=int)
    parser.add_argument("-top_k", default=3, type=int)
    parser.add_argument("-expert_dim", default=256, type=int)
    parser.add_argument("-router_loss_weight", default=1.0, type=float)
    parser.add_argument("-aux_loss_weight", default=0.01, type=float)

    args = parser.parse_args()

    max_experts = {
        'joint': 20,
        'group': 7,
        'exercise': config.CLASS_NUM,
    }[args.expert_type]
    if not 1 <= args.top_k <= max_experts:
        parser.error(
            "-top_k must be in [1, {}] for expert_type={}".format(
                max_experts, args.expert_type))
    if args.expert_dim <= 0:
        parser.error("-expert_dim must be positive")
    if args.tcn_channels <= 0:
        parser.error("-tcn_channels must be positive")
    if args.aux_loss_weight < 0:
        parser.error("-aux_loss_weight must be non-negative")
    if args.router_loss_weight < 0:
        parser.error("-router_loss_weight must be non-negative")

    # Embedder MLP layer 수 뒤에 BERT encoder layer 수 연결
    config.FILE_NAME = (
        f'{config.FILE_NAME}:{args.ext_layers}]'
    )
    moe_tags = {
        'joint': '[Joint_MoE]',
        'group': '[Group_MoE]',
        'exercise': '[Exercise_MoE]',
    }
    if args.moe:
        config.FILE_NAME += moe_tags[args.expert_type]
        config.FILE_NAME += '[Expert_Mode_{}]'.format(
            args.expert_mode.upper())
        if args.expert_mode == 'tcn':
            config.FILE_NAME += '[TCN_{}_C{}]'.format(
                args.tcn_conv_type.upper(), args.tcn_channels)
    if args.moe and args.expert_type in ('joint', 'group'):
        config.FILE_NAME += '[Expert_BERT_OUTPUT]'

    moe_slack_names = {
        'joint': 'Joint_MoE',
        'group': 'Group_MoE',
        'exercise': 'Exercise_MoE',
    }
    if not args.moe:
        moe_slack_name = 'No_MoE'
        expert_mode_slack_name = 'Not_Used'
        tcn_slack_name = 'TCN:Not_Used'
        embedding_slack_name = 'Exercise_Embedding:Not_Used'
        router_slack_name = 'Not_Used'
        top_k_slack_name = 'Top-K:Not_Used'
        expert_input_slack_name = 'Expert_Input:Not_Used'
    else:
        moe_slack_name = moe_slack_names[args.expert_type]
        expert_mode_slack_name = args.expert_mode
        if args.expert_mode == 'tcn':
            tcn_slack_name = 'TCN:{} C:{}'.format(
                args.tcn_conv_type, args.tcn_channels)
        else:
            tcn_slack_name = 'TCN:Not_Used'
        embedding_slack_name = 'Exercise_Embedding:Not_Used'
        router_slack_name = 'CLS'
        top_k_slack_name = 'Top-K:{}'.format(args.top_k)
        expert_input_slack_name = 'Expert_Input:BERT_OUTPUT'
    slack_header = (
        '[{}] [{}] [Expert_Mode:{}] [{}] [{}] [Router:{}] [{}] [{}]'.format(
            SLACK_SERVER_NAME,
            moe_slack_name,
            expert_mode_slack_name,
            tcn_slack_name,
            embedding_slack_name,
            router_slack_name,
            top_k_slack_name,
            expert_input_slack_name,
        )
    )

    # 완성된 FILE_NAME으로 logging 디렉터리 생성
    RUN_LOGGING = start_run_logging(
        PROJECT_ROOT,
        logging_root="/storage/jysuh/HPEFORBERT_LOG",
        run_name=config.FILE_NAME,
        slack_header=slack_header,
    )

    if args.condition_pr_output is None:
        args.condition_pr_output = os.path.join(
            str(RUN_LOGGING.run_dir),
            'condition_pr',
        )
    if args.exercise_ap_roc_output is None:
        args.exercise_ap_roc_output = os.path.join(
            str(RUN_LOGGING.run_dir),
            'exercise_ap_roc',
        )
    #
    fix_seed(args.seed)
    config.GPUS = args.device_id
    #
    args.world_size = 0
    print('NUM of ENCoder Layer:{}'.format(args.ext_layers))
    print('lr:{}'.format(args.lr))
    print('load bert ckpt:{}'.format(bool(args.train_from)))
    print('weighted_loss_value:{}'.format(args.weighted_loss_value))
    print('cond_loss_weight:{}'.format(args.cond_loss_weight))
    print('ex_loss_weight: {}'.format(args.ex_loss_weight))
    print('threshold:{}'.format(args.threshold))
    print('attach_cls_token_to_end_of_seqlen:{}'.format(args.attach_cls_token_to_end_of_seqlen))
    print('moe:{}'.format(args.moe))
    print('expert_type:{}'.format(args.expert_type))
    print('expert_mode:{}'.format(args.expert_mode))
    print('tcn_conv_type:{}'.format(args.tcn_conv_type))
    print('tcn_channels:{}'.format(args.tcn_channels))
    print('top_k:{}'.format(args.top_k))
    print('expert_dim:{}'.format(args.expert_dim))
    print('router_loss_weight:{}'.format(args.router_loss_weight))
    print('aux_loss_weight:{}'.format(args.aux_loss_weight))
    print('####### CONFIG #######')
    print()
    if RUN_LOGGING is not None:
        init_logger()
    elif args.log_file is not None:
        init_logger(args.log_file)

    if args.device_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.device_id))
        torch.cuda.set_device(args.device_id)
    config.DEVICE = device
    if RUN_LOGGING is not None:
        RUN_LOGGING.save_configuration(args, config, device=device)
        print('Run directory: {}'.format(args.run_dir))
        print('Training log: {}'.format(args.log_file))
        print('Checkpoint directory: {}'.format(args.multi_cls_model_path))




    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, args.device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, args.device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, args.device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, args.device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train' or args.mode == 'train-valid'):
            # config.TASK_MODE = 'TRAIN'
            train_ext(args, config, args.device_id)
        elif (args.mode == 'validate'):
            # config.TASK_MODE = 'VAL'
            validate_ext(args, config, args.device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, args.device_id, cp, step).to(device)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, args.device_id, cp, step)
