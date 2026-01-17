from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        has_score = "chosen_score" in features[0]

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                    **({"score": feature["chosen_score"]} if has_score else {}),
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                    **({"score": feature["rejected_score"]} if has_score else {}),
                }
            )

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        output = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }

        if has_score:
            output["score"] = batch["score"]

        return output


import torch
import torch.nn as nn
from base_trainer import RewardTrainer as BaseRewardTrainer


class RewardTrainer(BaseRewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop("loss_type", "bt")
        super().__init__(**kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )["logits"]

        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1

        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        has_score = "score" in inputs

        if self.loss_type != "bt" and not has_score:
            raise ValueError(
                f"loss_type='{self.loss_type}' requires chosen_score/rejected_score, "
                f"but dataset does not provide them."
            )

        if self.loss_type == "bt":
            # ===== Bradley–Terry =====
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()

        else:
            scores = inputs["score"]
            chosen_scores = scores[jidx].view(-1, 1)
            rejected_scores = scores[kidx].view(-1, 1)

            if self.loss_type == "margin":
                loss = -nn.functional.logsigmoid(
                    rewards_j - rewards_k - (chosen_scores - rejected_scores)
                ).mean()

            elif self.loss_type == "steerlm":
                loss = (
                    (rewards_j - chosen_scores) ** 2
                    + (rewards_k - rejected_scores) ** 2
                ).mean()

            elif self.loss_type == "lsam":
                logit_diff = rewards_j - rewards_k
                p = torch.sigmoid(logit_diff)
                alpha = torch.sigmoid(chosen_scores - rejected_scores)

                loss = -(
                    alpha * torch.log(p + 1e-8) + (1 - alpha) * torch.log(1 - p + 1e-8)
                ).mean()

            elif self.loss_type == "margin-aware":
                loss = (
                    (rewards_j - rewards_k - (chosen_scores - rejected_scores))
                    .pow(2)
                    .mean()
                )

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if return_outputs:
            return loss, {
                "rewards_j": rewards_j,
                "rewards_k": rewards_k,
            }

        return loss
