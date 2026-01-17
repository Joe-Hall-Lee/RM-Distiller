from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import torch
import torch.nn.functional as F
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
from utils import get_trainable_weights


@dataclass
class JointDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 处理 RM 训练部分：chosen 和 rejected
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                    "score": feature['chosen_score']
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                    "score": feature['rejected_score'],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # pad labels for RM part
        paded_length = batch["input_ids"].shape[1]
        label_paded = []
        for feature in features:
            label_chosen_paded = torch.tensor(
                feature["label_chosen"].tolist()
                + [self.label_pad_token_id]
                * (paded_length - len(feature["label_chosen"])),
                dtype=torch.int64,
            )
            label_rejected_paded = torch.tensor(
                feature["label_rejected"].tolist()
                + [self.label_pad_token_id]
                * (paded_length - len(feature["label_rejected"])),
                dtype=torch.int64,
            )
            label_paded.extend(
                [label_chosen_paded.view(
                    1, -1), label_rejected_paded.view(1, -1)]
            )
        label_paded = torch.concatenate(label_paded, dim=0)

        # 处理 SFT 部分：需要复制成 2 倍以匹配 RM 的维度
        sft_features = []
        for feature in features:
            # 每个样本复制 2 次，对应 chosen 和 rejected
            for _ in range(2):
                sft_features.append(
                    {
                        "input_ids": feature["student_teacher_input_ids"],
                        "attention_mask": feature["student_teacher_attention_mask"],
                    }
                )

        sft_batch = self.tokenizer.pad(
            sft_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # pad SFT labels
        sft_paded_length = sft_batch["input_ids"].shape[1]
        sft_labels_paded = []
        for feature in features:
            sft_label_paded = torch.tensor(
                feature["student_teacher_labels"].tolist()
                + [self.label_pad_token_id]
                * (sft_paded_length - len(feature["student_teacher_labels"])),
                dtype=torch.int64,
            )
            # 每个 label 复制 2 次
            sft_labels_paded.extend(
                [sft_label_paded.view(1, -1), sft_label_paded.view(1, -1)])
        sft_labels_paded = torch.concatenate(sft_labels_paded, dim=0)

        # 处理蒸馏部分：teacher response
        teacher_features = []
        teacher_prefix_lengths = []
        student_prefix_lengths = []

        for feature in features:
            # 每个样本复制 2 次
            for _ in range(2):
                teacher_features.append(
                    {
                        "input_ids": feature["teacher_input_ids"],
                        "attention_mask": feature["teacher_attention_mask"],
                    }
                )
                teacher_prefix_lengths.append(feature["teacher_prefix_len"])
                student_prefix_lengths.append(feature["student_prefix_len"])

        teacher_batch = self.tokenizer.pad(
            teacher_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids": batch["input_ids"],  # shape: [2B, seq_len]
            "attention_mask": batch["attention_mask"],  # shape: [2B, seq_len]
            "return_loss": True,
            "label": label_paded,  # shape: [2B, seq_len]
            "score": batch["score"],  # shape: [2B]
            # SFT 相关字段
            # shape: [2B, seq_len]
            "student_teacher_input_ids": sft_batch["input_ids"],
            # shape: [2B, seq_len]
            "student_teacher_attention_mask": sft_batch["attention_mask"],
            "student_teacher_labels": sft_labels_paded,  # shape: [2B, seq_len]
            # 蒸馏相关字段
            # shape: [2B, seq_len]
            "teacher_input_ids": teacher_batch["input_ids"],
            # shape: [2B, seq_len]
            "teacher_attention_mask": teacher_batch["attention_mask"],
            "teacher_prefix_len": teacher_prefix_lengths,  # length: 2B
            "student_prefix_len": student_prefix_lengths,  # length: 2B
        }
        return batch


class JointRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):

        # RM 训练相关参数
        self.info_to_save = kwargs.pop("info_to_save", {})

        # 蒸馏相关参数
        self.teacher_model = kwargs.pop("teacher_model", None)

        self.temperature = kwargs.pop("temperature", 1.0)
        self.eps = 1e-9

        # 损失权重
        self.reward_weight = kwargs.pop("reward_weight", 1.0)
        self.sft_weight = kwargs.pop("sft_weight", 1.0)
        self.kl_weight = kwargs.pop("kl_weight", 1.0)

        self.label_pad_token_id = -100
        # 初始化全局步数计数器
        self.global_step = 0

        super(JointRewardTrainer, self).__init__(**kwargs)

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.processing_class = self.tokenizer
        else:
            self.processing_class = None

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_mask = labels != self.label_pad_token_id

        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        logps = (per_token_logps * loss_mask).sum(-1)   # sum over tokens
        token_counts = loss_mask.sum(-1).clamp(min=1)
        logps = logps / token_counts                # average per token
        return logps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 准备和拼接所有输入
        batch_size = inputs["input_ids"].shape[0] // 2
        jidx = torch.arange(0, batch_size * 2, 2,
                            device=inputs["input_ids"].device)

        # 将 RM 输入和 SFT/KL 输入拼接成一个大张量
        # 形状: [2B (RM) + 2B (SFT/KL)] = [4B, seq_len]
        all_input_ids = torch.cat([
            inputs["input_ids"],
            inputs["student_teacher_input_ids"]
        ], dim=0)

        all_attention_mask = torch.cat([
            inputs["attention_mask"],
            inputs["student_teacher_attention_mask"]
        ], dim=0)

        # 执行唯一的一次前向传播
        # ----------------------------
        # 这个前向传播会同时计算 RM 的 rewards 和 SFT/KL 的 logits
        all_logits, _, all_rewards = model(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask
        )

        # 分输出并计算各个损失
        # --------------------------------

        # ========== RM Loss ==========
        # RM 的 rewards 在输出的前 2B 个元素中
        rm_rewards = all_rewards[:batch_size * 2]
        kidx = jidx + 1
        # reward_loss = - \
        #     torch.nn.functional.logsigmoid(
        #         rm_rewards[jidx] - rm_rewards[kidx]).mean()
        scores = inputs["score"]
        chosen_scores = scores[jidx]
        rejected_scores = scores[kidx]
        chosen_score = torch.tensor(
            chosen_scores, device=chosen_scores.device).view(-1, 1)
        rejected_score = torch.tensor(
            rejected_scores, device=rejected_scores.device).view(-1, 1)

        reward_loss = ((rm_rewards[jidx] - rm_rewards[kidx] -
                       (chosen_score - rejected_score)) ** 2).mean()

        # SFT 和 KL 的 logits 在输出的后 2B 个元素中
        sft_kl_logits = all_logits[batch_size * 2:]

        # ========== SFT Loss ==========
        sft_loss = torch.tensor(0.0, device=all_rewards.device)
        if self.sft_weight > 0:
            # SFT 损失只针对 "chosen" 对应的样本计算（即 jidx）
            sft_logits = sft_kl_logits[jidx]
            sft_logps = self.get_batch_logps(
                sft_logits,
                inputs["student_teacher_labels"][jidx]
            )
            sft_loss = -sft_logps.mean()
        # ========== KL Loss ==========
        kl_loss = torch.tensor(0.0, device=all_rewards.device)
        if self.kl_weight > 0 and self.teacher_model is not None:
            # 学生模型的 logits 已经从上面的大切分中得到了，无需再次计算
            student_logits_for_kl = sft_kl_logits[jidx]

            # 教师模型的前向传播
            with torch.no_grad():
                teacher_output = self.teacher_model(
                    input_ids=inputs["teacher_input_ids"][jidx],
                    attention_mask=inputs["teacher_attention_mask"][jidx]
                )
                teacher_logits = teacher_output.logits

            T = float(self.temperature)
            kl_losses = []

            pad_token_id = self.processing_class.pad_token_id
            for b in range(batch_size):
                teacher_prefix_len = inputs["teacher_prefix_len"][jidx[b]]
                student_prefix_len = inputs["student_prefix_len"][jidx[b]]

                # 计算可对齐长度
                teacher_available_len = teacher_logits.shape[1] - \
                    teacher_prefix_len - 1
                student_available_len = student_logits_for_kl.shape[1] - \
                    student_prefix_len - 1
                min_len = min(teacher_available_len, student_available_len)

                if min_len <= 0:
                    continue

                # 截取对应片段
                teacher_slice = teacher_logits[
                    b, teacher_prefix_len:teacher_prefix_len + min_len, :
                ]
                student_slice = student_logits_for_kl[
                    b, student_prefix_len:student_prefix_len + min_len, :
                ]

                # 计算 KL 散度
                t_log_prob = F.log_softmax(teacher_slice / T, dim=-1)
                s_log_prob = F.log_softmax(student_slice / T, dim=-1)
                t_prob = torch.exp(t_log_prob)

                per_token_kl = (t_prob * (t_log_prob - s_log_prob)).sum(dim=-1)

                # 使用 labels 进行 mask
                student_teacher_tokens = inputs["student_teacher_input_ids"][
                    jidx[b], student_prefix_len:student_prefix_len + min_len
                ]
                kl_mask = (student_teacher_tokens != pad_token_id).to(
                    dtype=per_token_kl.dtype
                )

                total_nonpad = kl_mask.sum()

                if total_nonpad.item() > 0:
                    sample_kl_loss = (per_token_kl * kl_mask).sum() / \
                        (total_nonpad + self.eps)
                    kl_losses.append(sample_kl_loss * (T * T))

            if kl_losses:
                kl_loss = torch.stack(kl_losses).mean()
        # ========== 总损失 ==========
        total_loss = (
            self.reward_weight * reward_loss
            + self.sft_weight * sft_loss
            + self.kl_weight * kl_loss
        )

        if return_outputs:
            return total_loss, {
                "reward_loss": reward_loss,
                "sft_loss": sft_loss,
                "kl_loss": kl_loss,
                "total_loss": total_loss,
            }
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            logits, _, rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            logps = self.get_batch_logps(logits, inputs["label"])

        return (None, logps.reshape(-1, 2), rewards.reshape(-1, 2))

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            # add config
            model.config.vhead_layer_type = self.info_to_save['layer_type']
            model.config.vhead_num_neurons = self.info_to_save['num_neurons']
            model.config.vhead_num_layers = self.info_to_save['num_layers']
            state_dict = get_trainable_weights(model)
            model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(output_dir)
