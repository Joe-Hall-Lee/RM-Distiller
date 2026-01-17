import sys
from copy import deepcopy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.aggregation import MeanMetric
from composer import ComposerModel
from composer.metrics import LanguageCrossEntropy
from composer.utils import dist, get_device
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
from llmfoundry.utils.config_utils import update_batch_size_info
from llmfoundry.models.hf import prepare_hf_model_for_fsdp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from train.cloud.train.utils import train, build_causal_lm
from train.cloud.train.data import build_feedback_dataloader

from train.cloud.model import (CLoudRewardModelConfig, CLoudRewardModel, COT_PROMPT)

log = logging.getLogger(__name__)


def convert_model_to_dtype(model, dtype=torch.bfloat16):
    """
    递归地将模型的所有参数和缓冲区转换为指定的 dtype
    """
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer_name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype)
    return model


def check_model_dtypes(model, model_name="Model"):
    """
    检查模型中所有参数的 dtype，用于调试
    """
    dtype_params = {}
    for name, param in model.named_parameters():
        dtype = str(param.dtype)
        if dtype not in dtype_params:
            dtype_params[dtype] = []
        dtype_params[dtype].append(name)
    
    print(f"=== {model_name} Parameter DTYPEs ===")
    for dtype, params in dtype_params.items():
        print(f"\n{dtype}: {len(params)} parameters")
        for p in params[:3]:
            print(f"  - {p}")
        if len(params) > 3:
            print(f"  ... and {len(params) - 3} more")
    print("=" * 50)


class ComposerRewardModel(ComposerModel):

    def __init__(self, reward_model_cfg, tokenizer, lm_weight, device):
        super().__init__()

        load_path = (
            reward_model_cfg.get("finetuned_model_name_or_path", reward_model_cfg.base_model_name_or_path)
        )
        reward_base_model = build_causal_lm(load_path, fsdp_config=reward_model_cfg.get("fsdp_config", None))
        self.reward_model = CLoudRewardModel(
            CLoudRewardModelConfig(
                feedback_method=reward_model_cfg.feedback_method,
                base_model_name_or_path=reward_model_cfg.base_model_name_or_path,
            ),
            reward_base_model
        )
        
        # === 关键修复：统一所有参数为 bfloat16 ===
        log.info("Converting all model parameters to bfloat16 for FSDP compatibility")
        self.reward_model = convert_model_to_dtype(self.reward_model, torch.bfloat16)
        
        # 检查 dtype（调试用，可以在确认无误后注释掉）
        check_model_dtypes(self.reward_model, "RewardModel")
        # ==========================================
        
        self.vocab_size = self.reward_model.reward_base_model.config.vocab_size
        self.feedback_method = self.reward_model.config.feedback_method
        self.tokenizer = tokenizer
        self.lm_weight = lm_weight

        self.train_metrics = {"PairwiseLoss": MeanMetric(), "PairwiseAcc": Accuracy(task="binary")}
        if self.feedback_method == "teacher":
            self.train_metrics["LMLoss"] = LanguageCrossEntropy()
        self.val_metrics = deepcopy(self.train_metrics)

        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.prepare_inner_model(self.reward_model, device)
        self.lm_loss._fsdp_wrap = False

    ###############
    # FSDP config
    ###############
    @staticmethod
    def prepare_inner_model(
        model,
        init_device,
    ):
        """Prepare the inner model for FSDP wrapping.

        Args:
            model: The model to prepare.
            init_device: The device to initialize the model on.
        """
        prepare_hf_model_for_fsdp(model.reward_base_model, init_device)

        model.fsdp_wrap_fn = model.reward_base_model.fsdp_wrap_fn
        model.activation_checkpointing_fn = model.reward_base_model.activation_checkpointing_fn

        del model.reward_base_model.fsdp_wrap_fn
        del model.reward_base_model.activation_checkpointing_fn

        model.param_init_fn = lambda module: model._init_weights(module)
        model.reward_base_model.param_init_fn = lambda module: model.reward_base_model._init_weights(module)

    ###############
    # Model calls
    ###############

    def forward(self, batch):

        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        rewards, logits = self.reward_model(input_ids, attention_mask)

        chosen_reward = rewards[:len(chosen_input_ids)]
        chosen_logits = logits[:len(chosen_input_ids)]
        rejected_reward = rewards[len(chosen_input_ids):]
        rejected_logits = logits[len(chosen_input_ids):]

        return {
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits
        }

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs else self.forward(batch)

    ##########
    # Losses
    ##########

    def compute_reward_loss(self, outputs, batch):
        reward_loss = -F.logsigmoid(outputs["chosen_reward"] - outputs["rejected_reward"])
        return reward_loss

    def compute_lm_loss(self, outputs, batch):
        return self.lm_loss(
            torch.cat([outputs["chosen_logits"], outputs["rejected_logits"]], dim=0).view(-1, self.vocab_size),
            torch.cat([batch["chosen_lm_labels"], batch["rejected_lm_labels"]], dim=0).view(-1)
        )

    def loss(self, outputs, batch):
        reward_loss = self.compute_reward_loss(outputs, batch)
        if self.feedback_method == "teacher":
            lm_loss = self.compute_lm_loss(outputs, batch)
            return reward_loss + self.lm_weight * lm_loss
        return reward_loss

    ##########
    # Metrics
    ##########

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanMetric):
            metric.update(self.compute_reward_loss(outputs, batch))

        elif isinstance(metric, BinaryAccuracy):
            batch_size = batch["chosen_input_ids"].shape[0]
            device = batch["chosen_input_ids"].device
            pairwise_preds = (outputs["chosen_reward"] > outputs["rejected_reward"]).long()
            pairwise_label = torch.ones((batch_size, 1), dtype=torch.long, device=device)
            metric.update(pairwise_preds, pairwise_label)

        elif isinstance(metric, LanguageCrossEntropy):
            metric.update(
                torch.cat([outputs["chosen_logits"], outputs["rejected_logits"]], dim=0).view(-1, self.vocab_size),
                torch.cat([batch["chosen_lm_labels"], batch["rejected_lm_labels"]], dim=0).view(-1)
            )


class ComposerSFTModel(ComposerModel):

    def __init__(self, model_cfg, tokenizer, device):
        super().__init__()

        model = build_causal_lm(model_cfg.base_model_name_or_path)
        self.model = model
        
        # === 关键修复：统一所有参数为 bfloat16 ===
        log.info("Converting all model parameters to bfloat16 for FSDP compatibility")
        self.model = convert_model_to_dtype(self.model, torch.bfloat16)
        # ==========================================
        
        self.vocab_size = self.model.config.vocab_size
        self.tokenizer = tokenizer
        self.feedback_method = model_cfg.feedback_method

        self.train_metrics = {"LMLoss": LanguageCrossEntropy()}
        self.val_metrics = deepcopy(self.train_metrics)

        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.prepare_inner_model(self.model, device)
        self.lm_loss._fsdp_wrap = False

    ###############
    # FSDP config
    ###############
    @staticmethod
    def prepare_inner_model(
        model,
        init_device,
    ):
        """Prepare the inner model for FSDP wrapping.

        Args:
            model: The model to prepare.
            init_device: The device to initialize the model on.
        """
        prepare_hf_model_for_fsdp(model, init_device)
        model.param_init_fn = lambda module: model._init_weights(module)

    ###############
    # Model calls
    ###############

    def forward(self, batch):

        input_ids = batch["chosen_input_ids"]
        attention_mask = batch["chosen_attention_mask"]

        if self.feedback_method == "csft":
            input_ids = torch.cat([input_ids, batch["rejected_input_ids"]], dim=0)
            attention_mask = torch.cat([attention_mask, batch["rejected_attention_mask"]], dim=0)

        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        logits = outputs.logits

        return {
            "logits": logits
        }

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs else self.forward(batch)

    ##########
    # Losses
    ##########

    def loss(self, outputs, batch):

        labels = batch["chosen_lm_labels"]
        if self.feedback_method == "csft":
            labels = torch.cat([labels, batch["rejected_lm_labels"]], dim=0)

        return self.lm_loss(
            outputs["logits"].view(-1, self.vocab_size),
            labels.view(-1)
        )

    ##########
    # Metrics
    ##########

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, LanguageCrossEntropy):

            labels = batch["chosen_lm_labels"]
            if self.feedback_method == "csft":
                labels = torch.cat([labels, batch["rejected_lm_labels"]], dim=0)

            metric.update(
                outputs["logits"].view(-1, self.vocab_size),
                labels.view(-1)
            )


def main(cfg):
    cfg = update_batch_size_info(cfg)

    # Initialize distributed training
    device = get_device(None)
    dist.initialize_dist(device, timeout=cfg.dist_timeout)

    # Build the model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model_name_or_path,
    )
    if tokenizer.chat_template is None:
        print("Chat template is none, overriding...")
        ref_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.chat_template = ref_tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.model.feedback_method in ["vanilla", "teacher", "oracle"]:
        model = ComposerRewardModel(cfg.model, tokenizer, cfg.lm_weight, device)
    elif cfg.model.feedback_method == "csft":
        model = ComposerSFTModel(cfg.model, tokenizer, device)

    # Build the training and eval datasets
    train_reward_loader = build_feedback_dataloader(
        cfg.train_loader,
        cfg.device_train_batch_size,
        tokenizer,
        cfg.model.feedback_method,
        COT_PROMPT
    )

    train(cfg, model, train_reward_loader, None)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)