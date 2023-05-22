import os
from datetime import datetime
from multiprocessing import Pool

# train config 
finetune_config = {
  "--model_name_or_path": "bert-base-cased",
  "--task_name": "mrpc",
  "--do_train": None,
  "--do_eval": None,
  "--max_seq_length": 128,
  "--per_device_train_batch_size": 32,
  "--learning_rate": 2e-5,
  "--num_train_epochs": 5,
  "--output_dir": "runs/mrpc/",
  "--evaluation_strategy": "steps"
}


# define command function
def train(device, config):
    cmd = f'CUDA_VISIBLE_DEVICES={device} python $HOME/code/transformers/examples/pytorch/text-classification/run_glue.py ' 
    options = []
    for k, v in config.items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']
    cmd += ' '.join(options)
    os.system(cmd)


def experiment_start(task,device,batch_size, lr, path, group):
    config = finetune_config
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    if task in ['cola', 'sst2', 'rte', 'mrpc']:
        epoch = 20
        config['--eval_steps'] = 50
    else:
        epoch =10
    
    config['--num_train_epochs'] = epoch
    config['--model_name_or_path'] = path
    config["--learning_rate"] = lr
    config["--task_name"] = task
    config['--per_device_train_batch_size'] = batch_size
    config["--output_dir"] = f'~/code/academic-budget-bert/runs/huggingface/{group}/{task}/{lr}_lr_{batch_size}bz'
    
    train(device, config)




# # loop over task    
# for task in ['rte']:
#     for lr in [2e-5]:
#         start_experiment(0, task, lr)
#     # run command

model_list = [
    ("models/teachers/bert-base-uncased", 'bert_base')
]



num_workers = 0


args = []

for model in model_list:

    for task in [ 'sst2', 'cola', 'rte', 'mrpc']: #'mnli', 'qqp','qnli',
    # for task in ['rte']:
    # for task in ["rte"]:
        
        idx = 0
        for lr in [1e-5, 3e-5, 5e-5, 8e-5]:
            for bz in [16, 32]:
                args.append((task, idx, bz, lr, model[0], model[1]))
                idx += 1
                num_workers +=1 

writer_workers = Pool(num_workers)
writer_workers.starmap(experiment_start, args)