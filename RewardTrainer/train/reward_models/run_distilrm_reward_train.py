from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
)
from RewardTrainer.train.reward_models.distilrm_reward_trainer import JointDataCollatorWithPadding, JointRewardTrainer
from RewardTrainer.train.reward_models.load_distilrm_datasets import load_joint_train_eval_dataset
from utils import print_trainable_parameters
from grm_utils import AutoModelForCausalLMWithValueHead


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_torch", metadata={"help": "The optimizer to use."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=1024)
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="")
    # data
    dataset: Optional[str] = field(
        default="RewardTrainer/data/skywork_10k_joint.json")
    dataset_mode: Optional[str] = field(
        default="joint", metadata={"help": "use 'joint' for joint training data"}
    )
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=10000)
    # model and loss
    base_model: Optional[str] = field(default="Qwen/Qwen2.5-3B-Instruct")
    teacher_model: Optional[str] = field(default="Qwen/Qwen3-14B")
    # log
    report_to: Optional[str] = field(
        default="none", metadata={"help": "use 'none', 'wandb'. "}
    )
    log_dir: Optional[str] = field(default="./reward_models_train")
    wandb_name: Optional[str] = field(
        default="joint_train",
    )
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=10000)
    debug: Optional[bool] = field(
        default=False, metadata={"help": "if debug=True, only train with 20 samples"}
    )
    # Joint training parameters
    reward_weight: Optional[float] = field(
        default=1.0, metadata={"help": "weight for reward model loss"}
    )
    sft_weight: Optional[float] = field(
        default=1.0, metadata={"help": "weight for SFT loss"}
    )
    kl_weight: Optional[float] = field(
        default=1.0, metadata={"help": "weight for KL divergence loss"}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "temperature for KL divergence"}
    )
    # Value head parameters
    layer_type: Optional[str] = field(default="mlp")  # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_joint_len{script_args.max_length}_fulltrain_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=1,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=1.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model, use_fast=False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token == None:
    if "Llama" in script_args.base_model:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load teacher tokenizer
teacher_tokenizer = None
if script_args.kl_weight > 0 and script_args.teacher_model:
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        script_args.teacher_model, use_fast=False
    )
    if teacher_tokenizer.pad_token == None:
        if "Llama" in script_args.teacher_model:
            teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Load datasets
train_dataset, eval_dataset = load_joint_train_eval_dataset(
    script_args.dataset,
    tokenizer,
    teacher_tokenizer,
    max_length=script_args.max_length,
    size=20 if script_args.debug else None,
)
print(
    "Training dataset size: {}, validation dataset size: {}".format(
        len(train_dataset), len(eval_dataset)
    )
)

model_params = {
    "vhead_layer_type": script_args.layer_type,
    "vhead_num_neurons": script_args.num_neurons,
    "vhead_num_layers": script_args.num_layers,
}
if len(script_args.attn_implementation):
    model_params["attn_implementation"] = script_args.attn_implementation

# Load teacher model if needed
teacher_model = None
if script_args.kl_weight > 0 and script_args.teacher_model:
    print(f"Loading teacher model: {script_args.teacher_model}")

    # 加载教师模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        script_args.teacher_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # teacher_model.resize_token_embeddings(len(tokenizer))
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

# Load student model
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.base_model, torch_dtype=torch.bfloat16, **model_params
)

# model.pretrained_model.resize_token_embeddings(len(tokenizer))
print_trainable_parameters(model)
model.config.pad_token_id = tokenizer.pad_token_id

# let value head trainable
if hasattr(model, "v_head"):
    for parameter in model.v_head.parameters():
        parameter.requires_grad = True

# Define the trainer parameters
trainer_params = {
    "model": model,
    "teacher_model": teacher_model,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "data_collator": JointDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
    "reward_weight": script_args.reward_weight,
    "sft_weight": script_args.sft_weight,
    "kl_weight": script_args.kl_weight,
    "temperature": script_args.temperature,
    "info_to_save": {
        "base_model": script_args.base_model,
        "layer_type": script_args.layer_type,
        "num_neurons": script_args.num_neurons,
        "num_layers": script_args.num_layers,
    },
}

# Train the model
trainer = JointRewardTrainer(**trainer_params)
print("joint training start")

trainer.train()
