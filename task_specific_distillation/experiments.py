# code inherited from 
# https://github.com/intersun/PKD-for-BERT-Model-Compression/blob/master/scripts/run_teacher_prediction.py
import os


import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from ruamel.yaml import YAML
from pathlib import Path
from datetime import datetime
from clearml import Task
import logging
import pandas as pd
from fairseq.models.roberta.model import RobertaModel
from os.path import expanduser
from clearml.backend_api import Session
import argparse 
import socket

Session._session_initial_timeout = (15., 30.)
home = expanduser("~")



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()



yaml = YAML(typ='rt')
yaml.preserve_quotes = True

def run_process(proc):
    os.system(proc)

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0


config_inter_format_path = 'config/format/inter_distill_format.yaml'
config_pred_format_path = 'config/format/pre_distill_format.yaml'

EXP_GROUP = 'task-specific-comparison'
PROJECT_NAME = 'master-thesis'

task_param = {
    'mnli': {
        'general':[3, 1e-5, 32, 123873, 7432, home + '/data/fairseq/MNLI-bin/', 'checkpoints/teacher/mnli_base'],
    }, 


    'qnli': {
        'general': [2, 1e-5, 32, 33112, 1986, home +'/data/fairseq/QNLI-bin', 'checkpoints/teacher/qnli_base'],
    },


    'qqp': {
        'general': [2, 1e-5, 32, 113272, 28318,home +'/data/fairseq/QQP-bin', 'checkpoints/teacher/qqp_base'],
    },


    'rte': { 
        'general':[2, 2e-5, 16, 2036, 122, home +'/data/fairseq/RTE-bin', 'checkpoints/teacher/rte_12'],
    },

    'sst-2': {
        'general':  [2, 1e-5, 32, 20935, 1256, home +'/data/fairseq/SST-2-bin', 'checkpoints/teacher/sst-2_12'],
    },
    'mrpc': {
        'general': [2, 1e-5, 16, 2296, 137, home +'/data/fairseq/MRPC-bin', 'checkpoints/teacher/mrpc_base' ],
    },

    'cola': {
        'general': [2, 1e-5, 16, 5336, 320, home +'/data/fairseq/CoLA-bin','checkpoints/teacher/cola_base' ],
    },
}


def task_specific_config(config, task, epoch):

    factor = int(epoch / 10)
    param = task_param[task]
    config['args']['--num-classes'] = param['general'][0]
    config['args']['--lr'] = param['general'][1]
    config['args']['--max-sentences'] = param['general'][2]
    config['args']['--total-num-update'] = param['general'][3] * factor
    config['args']['--warmup-updates'] = param['general'][4] * factor
    config['data_path'] = param['general'][5]
    config['args']['--data_name_or_path'] = param['general'][5]
    config['args']['--teacher_model_checkpoint'] = param['general'][6]


    return config





def config_specify_inter(task, method, layers, selection_list, epoch, seed, mapping):
    config = yaml.load(Path(config_inter_format_path))
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    log_path = f"checkpoints/runs/{current_time}_{seed}_{task}_{method}"
    config['args']['--mapping'] = mapping
    model_to_pass = f'{log_path}/checkpoint_last.pt'
    config['args']['--save-dir'] = log_path
    config['args']['--tensorboard-logdir'] = log_path

    config['args']['--feature_learn'] = method
    config = task_specific_config(config, task, epoch)

    config['args']['--encoder-layers'] = layers
    teacher_path = config['args']['--teacher_model_checkpoint']
    init_path = layer_weight_selection(teacher_path, task, selection_list)
    config['args']['--init_pretrained'] = init_path
    # config['args']['--init_pretrained'] = task_param[task]['init'][layers]
    
    config['args']['--max-epoch'] = epoch
    num_layers = config['args']['--encoder-layers']


    config['args']['--run-name'] =   task + '_' + method + '_' + 'inter' + f'{epoch}' + 'epoch' + '_' + f'{num_layers}' + 'S' + '_' + current_time + 'seed' + f'{seed}' + '_' +f'{mapping}'

    if method == 'crd':
        config['args']['--criterion'] = 'sentence_prediction_tiny'
        config['args']['--crd_weight'] = 1
        config['args']['--s_dim_feat'] = 3072
        config['args']['--t_dim_feat'] = 9984
    if method == 'kd':
        config['args']['--criterion'] = 'sentence_prediction_crd'
        config['args']['--crd_weight'] = 0
        config['args']['--kd_weight'] = 0.5
        config['args']['--temperature'] = 2
        config['args']['--run-name'] =   task + '_' + method + '_' + 'pred' + f'{epoch}' + 'epoch' + '_' + f'{num_layers}' + 'S' + '_' + current_time + 'seed' + f'{seed}' + f"_mapping_{mapping}"
    return config, model_to_pass

def config_specify_pred(task, method, model_to_pass, layers, epoch, seed, mapping):
    config = yaml.load(Path(config_pred_format_path))
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    log_path = f"checkpoints/runs/{current_time}_{seed}_{task}_{method}"
    config = task_specific_config(config, task, epoch)
    
    config['args']['--restore-file'] = model_to_pass
    config['args']['--save-dir'] = log_path
    config['args']['--tensorboard-logdir'] = log_path
    config['args']['--max-epoch'] = epoch
    config['args']['--run-name'] =  task + '_' + method + '_' + 'pred' + f'{epoch}' + 'epoch' + '_' + f'{layers}' + 'S' + '_' + current_time + 'seed' + f'{seed}' + f'{mapping}'
    config['args']['--encoder-layers'] = layers 

    return config, log_path



def train(config, device, seed, group):

    config['args']['--seed'] = seed
    
    cmd = f'CUDA_VISIBLE_DEVICES={device} python %s/train.py ' % PROJECT_FOLDER
    DATA_PATH = ' ' + config['data_path'] + ' '
    cmd += DATA_PATH

    config['args']['--project-name'] = PROJECT_NAME
    config['args']['--experiment-group'] = group

    options = []
    for k, v in config['args'].items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']


    cmd += ' '.join(options)
    
    log_path = config['args']['--save-dir']
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    config_save_path = log_path + '/config.yaml'
    yaml.dump(config, Path(config_save_path))
    os.system(cmd)

    return None







# TODO: Wrapp the two-stage experiments as function

def two_stage_training(task, method, student_layer, selection, epoch, seed, device, mapping, group):
    selection_list = [int(item) for item in selection.split(',')]
    config, model_to_pass = config_specify_inter(task, method, student_layer, selection_list, epoch, seed, mapping)
    config['args']['--layer_selection'] = selection
    
    train(config, device, seed, group)

    logger.info(f'{method}, {task}, Intermediate layer training finished.')


    config, save_dir = config_specify_pred(task, method, model_to_pass, student_layer, epoch, seed, mapping)
    
    train(config, device, seed, group)

    logger.info(f'{method}, {task}, Prediction layer trianing finished.')

def one_stage_training(task, method, student_layer, selection, epoch, seed, device, mapping, group):
    selection_list = [int(item) for item in selection.split(',')]
    config, model_to_pass = config_specify_inter(task, method, student_layer, selection_list, epoch, seed, mapping)
    config['args']['--layer_selection'] = selection
    train(config, device, seed, group)


def pred_continue_training(model_to_pass,task, method, student_layer, selection, epoch, seed, device, mapping, group):
    selection_list = [int(item) for item in selection.split(',')]
    config, save = config_specify_pred(task, method, model_to_pass, student_layer, epoch, seed, mapping)
    train(config, device, seed, group)

def layer_weight_selection(teacher_path, task, selection_list):
    teacher_model = RobertaModel.from_pretrained(teacher_path, checkpoint_file='checkpoint_best.pt', data_name_or_path=task_param[task]['general'][5])
    state_dict = teacher_model.state_dict()
    teacher_state_dict={}
    for k, v in state_dict.items():
        if 'model.' in k:
            name = k[6 :] # remove `module.`
            teacher_state_dict[name] = v
        elif '_float_tensor' in k:
            continue
        else:
            teacher_state_dict[k] = v
    std_state_dict = {}
    # Embedding
    for w in ['embed_tokens', "embed_positions"]:
        std_state_dict[f"decoder.sentence_encoder.{w}.weight"] = teacher_state_dict[f"decoder.sentence_encoder.{w}.weight"]
    for w in ['weight', 'bias']:
        std_state_dict[f"decoder.sentence_encoder.emb_layer_norm.{w}"] = teacher_state_dict[f"decoder.sentence_encoder.emb_layer_norm.{w}"]

    # Transformer Blocks#
    std_idx = 0
    # select teacher layers 

    for teacher_idx in selection_list:
        for layer in ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj',
        'self_attn.out_proj','self_attn_layer_norm', 'fc1','fc2','final_layer_norm',
        ]:
            for w in ['weight', 'bias']:
                std_state_dict[f"decoder.sentence_encoder.layers.{std_idx}.{layer}.{w}"] = teacher_state_dict[
                    f"decoder.sentence_encoder.layers.{teacher_idx}.{layer}.{w}"
                ]
        std_idx += 1

    # LM Head
    for layer in ['lm_head', 'lm_head.dense', 'lm_head.layer_norm']:
        for w in ['weight', 'bias']:
            std_state_dict[f"decoder.{layer}.{w}"] = teacher_state_dict[f"decoder.{layer}.{w}"]
    
    path = teacher_path + '/' + task + '_' + '_'.join([str(x) for x in selection_list]) + '.pt'
    if not os.path.exists(path):
        torch.save(std_state_dict, path)
    return path



def pred_continue(task, method, device, stage, mapping, mig_path,model_to_pass_dic_path, group, seeds):
    model_to_pass_list_all = yaml.load(Path(model_to_pass_dic_path))
    model_to_pass_list = model_to_pass_list_all[method]
    seed_list = [int(item) for item in seeds.split(',')]
    if socket.gethostname() == 'grancir':
        mig_list = yaml.load(Path(mig_path))

    # idx = 0
    # task_list = ['qqp']
    # method_list  = ['att_mse_hidden_mse_learn', 'att_kl_hiden_mse_learn', 'att_kl_learn', 'minilm', "hidden_mse_learn" , 'att_mse_learn' ]
    student_layer=3
    selection_list = '3,7,11'
    epoch=10
    # for task in task_list:
    
    args = []

    num_workers = 0
    # mig_key = f'0/{idx}'
    idx = 0
    # device = 0
    # model_to_pass_list = model_to_pass_dic[task]
    for seed in seed_list:
        if socket.gethostname() == 'grancir':
            print('Using grancir GPUs')
            device = mig_list[idx]
        else:
            device = idx
        model_to_pass = model_to_pass_list[idx]
        args.append((model_to_pass,task, method, student_layer, selection_list, epoch, seed, device, mapping, group))
        num_workers += 1
        idx += 1
        

    writer_workers = Pool(num_workers)
    writer_workers.starmap(pred_continue_training, args)


def main_task_method(task, method, device, stage, selection ,mapping, mig_path, group, seeds):
    seed_list = [int(item) for item in seeds.split(',')]
    # seed_list = [1]#,2,3,4]
    if socket.gethostname() == 'grancir':
        mig_list = yaml.load(Path(mig_path))

    student_layer=3
    epoch=10
    
    args = []

    num_workers = 0

    idx = 0


    for seed in seed_list:
        if socket.gethostname() == 'grancir':
            print('Using grancir GPUs')
            device = mig_list[idx]
        else:
            device = idx
        args.append((task, method, student_layer, selection, epoch, seed, device, mapping, group))
        num_workers += 1
        idx += 1
        

    writer_workers = Pool(num_workers)
    if stage == 'two_stage':
        writer_workers.starmap(two_stage_training, args)
    else:
        writer_workers.starmap(one_stage_training, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, required=False)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, required=False, help='the gpu MIG device to use')
    parser.add_argument('-s', '--stage', type=str, required=True, help='one or two-stage training')
    parser.add_argument('--mapping', type=str, required=True)
    parser.add_argument('--mig_path', type=str,required=False, default=None)
    parser.add_argument('--pass_dic', type=str, required=False, default=None)
    parser.add_argument('--init', type=str, required=False, default='3,7,11')
    parser.add_argument('--group', type=str, required=False, default='task-specific-comparison')
    parser.add_argument('--seeds', type=str, required=True, default='1')
    args = parser.parse_args()
    if args.experiment == 'method_specific':
        main_task_method(args.task, args.method, args.device, args.stage, args.init ,args.mapping, args.mig_path, args.group, args.seeds)
    if args.experiment == 'pred':
        pred_continue(args.task, args.method, args.device, args.stage, args.mapping, args.mig_path, args.pass_dic, args.group, args.seeds)
