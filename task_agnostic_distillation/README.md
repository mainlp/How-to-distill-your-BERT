# Task-Agnostic Distillation 
We build our task-agnostic distillation framework based on  [academic-budget-bert](https//github.com/IntelLabs/academic-budget-bert) which supports pre-training BERT-like models under an academic budget. 

## Data Preprocessing
[academic-budget-bert](https//github.com/IntelLabs/academic-budget-bert) provided a nice [`script`](dataset/) for chunking and preprocessing the Wikipedia and BookCorpus raw text data. 

## Distillation
Example
```bash
python -m torch.distributed.launch run_pretraining.py \
    --model_type bert-mlm \
    --deepspeed \
    --tokenizer_name bert-base-uncased \
    --hidden_act gelu \
    --hidden_dropout_prob 0.1 \
    --attention_probs_dropout_prob 0.1 \
    --encoder_ln_mode post-ln \
    --lr 5e-4 \
    --lr_schedule step \
    --max_steps 100000 \
    --curve linear \
    --warmup_proportion 0.06 \
    --gradient_clipping 0.0 \
    --optimizer_type adamw \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_eps 1e-6 \
    --dataset_path ~/data/budget/masked \
    --output_dir ~/code/academic-budget-bert/training-out \
    --print_steps 100 \
    --num_epochs_between_checkpoints 100 \
    --project_name budget-bert-pretraining-post \
    --validation_epochs 5 \
    --validation_epochs_begin 5 \
    --validation_epochs_end 5 \
    --validation_begin_proportion 0.01 \
    --validation_end_proportion 0.01 \
    --validation_micro_batch 64 \
    --data_loader_type dist \
    --do_validation \
    --use_early_stopping \
    --early_stop_time 180 \
    --early_stop_eval_loss 6 \
    --seed 43 \
    --layernorm_embedding  \
    --hidden_size 768 \
    --num_hidden_layers 6 \
    --num_attention_heads 12 \
    --intermediate_size 3072 \
    --layer_selection 11 \
    --distillation \
    --teacher_path models/teachers/bert-base-uncased \
    --total_training_time 90.0 \
    --early_exit_time_marker 90.0 \
    --train_batch_size 1024 \
    --train_micro_batch_size_per_gpu 256 \
    --job_name hidden_mse_init \
    --current_run_id 1 \
    --method hidden_mse
    --finetune_time_markers 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
    --fused_linear_layer false
    --layer_norm_type pytorch
    --fp16 
    --fp16_backend ds
    --student_initialize 
```

