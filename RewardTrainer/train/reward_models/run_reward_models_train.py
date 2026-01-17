from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from reward_trainer import RewardTrainer, RewardDataCollatorWithPadding
from load_datasets import load_train_eval_dataset
from utils import print_trainable_parameters


@dataclass
class ScriptArguments:
    # Training args
    per_device_train_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={
                                            "help": "The number of training epochs for the reward model."})
    optim: Optional[str] = field(default="adamw_torch",  metadata={
                                 "help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024)
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="")
    # Data
    dataset: Optional[str] = field(default='llm-blender/Unified-Feedback')
    dataset_mode: Optional[str] = field(default='', metadata={
                                        "help": "use from '', '40k', and '400k' for the paper's experiments"},)

    # Evaluation
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_steps: Optional[int] = field(default=100)
    # Model and loss
    base_model: Optional[str] = field(default="google/gemma-2b-it")
    loss_type: Optional[str] = field(default='bt', metadata={
                                     'help': "use 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    weight_ratio: Optional[float] = field(
        default=0.1, metadata={'help': 'the ratio for label smooth or posreg'})
    # Logging
    report_to: Optional[str] = field(default='none', metadata={
                                     'help': "use 'none', 'wandb'. "})
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="no")
    debug: Optional[bool] = field(default=False, metadata={
                                  'help': 'if debug=True, only train with 4 samples'})


# Parse the script arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.loss_type}_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"


training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_total_limit=0,
    save_safetensors=True,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=1,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False
)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model, use_fast=False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(
    script_args.dataset, tokenizer, mode=script_args.dataset_mode, size=4 if script_args.debug else None)
print('Training dataset size: {}, validation dataset size: {}'.format(
    len(train_dataset), len(eval_dataset)))

if len(script_args.attn_implementation):
    model_params = {
        "attn_implementation": script_args.attn_implementation,
    }
else:
    model_params = {}

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model, num_labels=1,
    torch_dtype=torch.bfloat16,
    **model_params
)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print_trainable_parameters(model)

# Data collator
data_collator = RewardDataCollatorWithPadding(
    tokenizer=tokenizer, max_length=script_args.max_length)

# Define the trainer parameters
trainer_params = {
    "model": model,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "data_collator": data_collator,
    'loss_type': script_args.loss_type,
}

trainer = RewardTrainer(**trainer_params)

print_trainable_parameters(trainer.model)

print('training start')
trainer.train()
# Save model
model.config.use_cache = True

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(
    trainer.model, StateDictType.FULL_STATE_DICT, save_policy
):
    trainer.save_model()
