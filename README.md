# How to Distill your BERT: An Empirical Study on the Impact of Weight Initialisation and Distillation Objectives

Initial code release for the paper:

**How to Distill your BERT: An Empirical Study on the Impact of Weight Initialisation and Distillation Objectives** (ACL 2023)

Xinpeng Wang, Leonie Weissweiler, Hinrich Schütze and Barbara Plank. 


## Task-Specific-Distillation

We inherit [Fairseq](https://github.com/facebookresearch/fairseq) framework for task-specific-distillation of the RoBERTa model.  

### Train 

run `task_specific_distillation/experiments.py` for task-specific distillation on RoBERTa model.
```bash
python experiments.py  --task {task} --method {method} -e {experiment} -s {stage} --mapping {mapping} --init {init} --group {group} --seeds {seeds} 
```

``task:  mnli, qnli, sst-2, cola, mrpc, qqp, rte ``

``method: kd, hidden_mse_learn, hidden_mse_token, crd, att_kl_learn, att_mse_learn``


## Task-Agnostic-Distillation 
The task-anostic-distillation code is based on the work [izsak-etal-2021-train](https://github.com/IntelLabs/academic-budget-bert). 

### Data Preperation

The [`dataset`](dataset/) directory includes scripts to pre-process the datasets we used in our experiments (Wikipedia, Bookcorpus). See dedicated [README](dataset/README.md) for full details.

### Pretrain

run `task_agnostic_distillation/experiments.py` for distilling a transformer model from BERT_large during the pre-training stage. 

```bash
python -m torch.distributed.launch run_pretraining.py --method {distillation_objective} --student_initialize ... 
```

See `task_agnostic_distillation/README.md` for a complete bash code example and detailed explanation of all the training configuration. 

### Finetuning

Run `task_agnostic_distillation/run_glue.py` for finetuning a saved checkpoint on GLUE tasks. 

example :

```bash
python run_glue.py \
  --model_name_or_path <path to model> \
  --task_name MRPC \
  --max_seq_length 128 \
  --output_dir /tmp/finetuning \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 50 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50
```

## Cite
```
TBA
```

