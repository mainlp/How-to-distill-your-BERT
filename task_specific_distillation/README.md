# Task-specific-distillation 
This folder contains code for task-specific-distillation framework based on Faiseq. 
For full explanation and technical support, please refer to the official [Fairseq repo](https://github.com/facebookresearch/fairseq).
## Data Preperation 

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Preprocess GLUE task data:
```bash
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.


## Distillation
We created a python script which helps to create and run bash script for running the `train.py` with different parameters such distillation method, GLUE task, mapping strategy, initialisation choices, experiment group (for clearML logging) and seeds list. 

Example:
```bash
python experiments.py  --task mnli --method att_mse_learn -e method_specific -s two_stage --mapping skip --init 0,1,2 --group experiment_1 --seeds 1,2,3,4
```

