# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizer

from .models import REWARD_MODEL_CONFIG


def format_conversation_for_judge(prompt):
    """
    将 prompt 转换为适合 Judge 评估的格式
    - 如果是字符串，直接返回
    - 如果是列表（多轮对话），格式化为可读的对话文本
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and len(prompt) > 0:

        if len(prompt) == 1 and isinstance(prompt[0], dict):
            return prompt[0].get("content", "")
        # 格式化多轮对话为易读文本
        conversation_parts = []
        for turn in prompt:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                # 格式化为 "User: xxx" 或 "Assistant: xxx"
                role_label = "User" if role == "user" else "Assistant"
                conversation_parts.append(f"{role_label}: {content}")

        return "\n\n".join(conversation_parts)

    return ""
# HuggingFace Hub locations
CORE_EVAL_SET = "data/rewardbench/filtered.json"
EXTRA_PREF_SETS = "allenai/pref-test-sets"
BON_CANDIDATES = "ai2-adapt-dev/HERM_BoN_candidates"  # private until officially supported


def torch_dtype_mapping(dtype_str):
    """
    Helper function for argparse to map string to torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"Invalid torch dtype: {dtype_str}")
    return dtype_map[dtype_str]


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


def map_conversations_testsets(example):
    prompt = example["prompt"]
    example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
    example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
    return example


def load_eval_dataset(
    core_set: bool = True,
    dataset: str = CORE_EVAL_SET,
    custom_dialogue_formatting: bool = False,
    conv = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "id"],
    return_extra_data: bool = False,
    max_turns: int = None,
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for RewardBench or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for RewardBench.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset.
        return_extra_data: return extra metadata for expanded logging (mostly in CLI)
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    raw_dataset = []
    with open(dataset, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                raw_dataset.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

    # Initialize an empty DatasetDict to hold the modified datasets
    modified_datasets = DatasetDict()

    # Iterate over each entry in the raw dataset
    for entry in raw_dataset:
        subset_name = entry["subset"]
        
        # Check if the subset already exists in the DatasetDict
        if subset_name not in modified_datasets:
            modified_datasets[subset_name] = []
        
        # Add the entry to the corresponding subset
        modified_datasets[subset_name].append(entry)

    # Convert the lists of dictionaries to Hugging Face Datasets
    for subset_name, subdataset in modified_datasets.items():
        modified_datasets[subset_name] = Dataset.from_dict({key: [entry[key] for entry in subdataset] for key in subdataset[0]})

    # Convert the DatasetDict to a list of Dataset objects
    dataset_list = list(modified_datasets.values())

    # Concatenate all the modified datasets into one dataset
    raw_dataset = concatenate_datasets(dataset_list)

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if logger is not None:
            logger.info("*** Preparing dataset with HF Transformers ***")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        dataset = raw_dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=8,
            load_from_cache_file=False,
        )

    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
                # 多轮对话
                prompt = example["prompt"]
                example["text_chosen"] = prompt + \
                    [{"role": "assistant", "content": example["chosen"]}]
                example["text_rejected"] = prompt + \
                    [{"role": "assistant", "content": example["rejected"]}]
            else:
                # 单轮对话
                example["text_chosen"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ]
                example["text_rejected"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"]},
                ]

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["text_chosen"]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset
    subsets = dataset["subset"]
    if return_extra_data:
        return dataset, subsets

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset, subsets


def load_bon_dataset(
    best_of: int = 16,
    custom_dialogue_formatting: bool = False,
    conv = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    remove_columns: List[str] = None,
):
    """
    Loads the BON candidates dataset.
    """

    alpaca_eval = load_dataset("ai2-adapt-dev/HERM_BoN_candidates", "alpaca_eval")
    mt_bench = load_dataset("ai2-adapt-dev/HERM_BoN_candidates", "mt_bench")
    merged_alpaca_eval = concatenate_datasets([alpaca_eval["zephyr"], alpaca_eval["tulu"]])
    merged_mt_bench = concatenate_datasets([mt_bench["zephyr"], mt_bench["tulu"]])

    # add column "subset" alpaca_eval
    merged_alpaca_eval = merged_alpaca_eval.add_column(
        "subset", ["alpaca_eval" for i in range(len(merged_alpaca_eval))]
    )
    # rename column dataset to dataset_details
    merged_alpaca_eval = merged_alpaca_eval.rename_column("dataset", "dataset_details")
    merged_mt_bench = merged_mt_bench.rename_column("category", "dataset_details")
    # convert alpaca eval id to int
    merged_alpaca_eval = merged_alpaca_eval.cast_column("id", Value(dtype="int64", id=None))

    # rename generator to model
    merged_alpaca_eval = merged_alpaca_eval.rename_column("generator", "model")
    merged_mt_bench = merged_mt_bench.rename_column("generator", "model")

    # rename instruction to prompt
    merged_alpaca_eval = merged_alpaca_eval.rename_column("instruction", "prompt")
    merged_mt_bench = merged_mt_bench.rename_column("instruction", "prompt")

    # add column "subset" mt_bench
    merged_mt_bench = merged_mt_bench.add_column("subset", ["mt_bench" for i in range(len(merged_mt_bench))])

    # remove question_id
    merged_mt_bench = merged_mt_bench.remove_columns("question_id")

    # remove model_id
    merged_mt_bench = merged_mt_bench.remove_columns("model_id")

    raw_dataset = concatenate_datasets([merged_alpaca_eval, merged_mt_bench])

    # unroll every row in ['output'] to a new row, all other columns are copied,
    # index is changed to tuple (index, output_index)
    def unroll_output(row, n):
        rows = []
        outputs = row["output"]
        id = row["id"]

        for i, output in enumerate(outputs[:n]):
            new_row = row.copy()
            new_row["output_new"] = output
            new_row["index"] = [id, i]
            del new_row["output"]
            del new_row["id"]
            rows.append(new_row)
        return rows

    new_dataset = []
    for row in raw_dataset:
        new_dataset.extend([r for r in unroll_output(row, n=best_of)])

    # create huggingface dataset through pandas
    unrolled_dataset = Dataset.from_pandas(pd.DataFrame(data=new_dataset))
    # rename output_new to text
    unrolled_dataset = unrolled_dataset.rename_column("output_new", "input")
    unrolled_dataset = unrolled_dataset.rename_column("index", "id")

    # Apply chat template
    usable_tokenizer = check_tokenizer_chat_template(tokenizer)

    # assert either conv is passed or tokenizer has chat_template
    assert conv is not None or usable_tokenizer

    if usable_tokenizer:
        if logger is not None:
            logger.info("*** Preparing dataset with HF Transformers ***")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        dataset = unrolled_dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer, "ift": True},
        )

    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations_ift(example):
            example["text"] = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["input"]},
            ]
            return example

        dataset = unrolled_dataset.map(
            map_conversations_ift,
            # fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    # remove column input
    dataset = dataset.remove_columns(remove_columns)

    return dataset


def prepare_dialogue_from_tokenizer(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    ift: bool = False,
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
            if messages[-1]["role"] != "user":
                print(example)
            # assert that the last message before this is user
            assert messages[-1]["role"] == "user"

            # required for DPO code only, otherwise discarded
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["chosen"]})
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages[-1] = {"role": "assistant", "content": example["rejected"]}
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
        # single turn
        else:
            # needed for DPO
            messages = [
                {"role": "user", "content": example["prompt"]},
            ]
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
    elif ift:
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["input"]},
            ]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def prepare_dialogue(
    example: Dict[str, Any],
    dialogue_template,
    ift: bool = False,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            dialogue_template.messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    dialogue_template.messages.append([dialogue_template.roles[0], p])
                else:
                    dialogue_template.messages.append([dialogue_template.roles[1], p])
            # assert that the last message before this is user
            assert dialogue_template.messages[-1][0] == dialogue_template.roles[0]

            # needed for DPO
            temp_prompt = dialogue_template.get_prompt()

            # end with chosen/rejected
            dialogue_template.messages.append([dialogue_template.roles[1], example["chosen"]])
            example["text_chosen"] = dialogue_template.get_prompt()

            dialogue_template.messages[-1] = [dialogue_template.roles[1], example["rejected"]]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt

        # single turn
        else:
            if isinstance(example["prompt"], list):
                example["prompt"] = example["prompt"][0]
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
            ]
            temp_prompt = dialogue_template.get_prompt()

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["chosen"]],
            ]
            example["text_chosen"] = dialogue_template.get_prompt()
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["rejected"]],
            ]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt
    elif ift:
        if isinstance(example["prompt"], list):
            example["prompt"] = example["prompt"][0]

        # get prompt
        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
        ]
        temp_prompt = dialogue_template.get_prompt()

        # get messages
        if "messages" in example:
            # convert to FastChat format (list of list)
            # original format:
            # [
            #     {"role": "user", "content": example["prompt"]},
            #     {"role": "assistant", "content": example["rejected"]},
            # ]
            dialogue_template.messages = []
            for i, line in enumerate(example["messages"]):
                role = dialogue_template.roles[0] if i % 2 == 0 else dialogue_template.roles[1]
                dialogue_template.messages.append([role, line["content"]])
        else:
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["input"]],
            ]
        example["text"] = dialogue_template.get_prompt()
        example["prompt"] = temp_prompt  # needed for DPO

    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def load_model_config(model_name):
    """
    Load the model for evaluation.
    """
    # if custom config, load that, else return default
    if model_name in REWARD_MODEL_CONFIG:
        return REWARD_MODEL_CONFIG[model_name]
    else:
        return REWARD_MODEL_CONFIG["default"]


def load_eval_dataset_multi(
    core_set: bool = True,
    dataset: str = None,  # alternate dataset
    custom_dialogue_formatting: bool = False,
    conv = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["texts_chosen", "texts_rejected", "id"],
    return_extra_data: bool = False,
    max_turns: int = None,
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for RewardBench 2 or a user-passed dataset, for running generative models

    Args:
        core_set: if True, load the core eval set for RewardBench 2.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset. Because of the intricacies of handling the Ties subset,
                we keep the "subset" and "num_correct" columns for RB2.
        return_extra_data: return extra metadata for expanded logging (mostly in CLI)
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    # consider making this force the -no-ties version of core eval set
    raw_dataset = pd.read_json(dataset, lines=True)
    raw_dataset = Dataset.from_pandas(raw_dataset)
    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = raw_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=8,
                load_from_cache_file=False,
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = raw_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv},
                num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
                load_from_cache_file=False,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
                example["prompt"] = format_conversation_for_judge(example["prompt"])
            chosen_texts = []
            for chosen_response in example["chosen"]:
                chosen_texts.append(
                    [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": chosen_response},
                    ]
                )
            example["texts_chosen"] = chosen_texts
            rejected_texts = []
            # multiple rejected responses
            for rejected_response in example["rejected"]:
                rejected_texts.append(
                    [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": rejected_response},
                    ]
                )
            example["texts_rejected"] = rejected_texts
            return example

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )
        logger.info(f"Dataset columns: {dataset.column_names}")

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["texts_chosen"][0]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns(
        [c for c in all_cols if c not in keep_columns])

    return dataset


def reroll_and_score_dataset(dataset, total_completions, cols_to_combine=["text", "scores"]):
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Validate that sum of total_completions matches dataset length
    if sum(total_completions) != len(df):
        raise ValueError(
            f"Sum of total_completions ({sum(total_completions)}) does not equal dataset length ({len(df)})"
        )

    rerolled_rows = []
    current_idx = 0

    # Process each group with its specified number of completions
    for group_size in total_completions:
        group = df.iloc[current_idx: current_idx + group_size]

        # Create new row
        new_row = {}
        # print(group['scores'])
        # Handle text and score columns - combine into lists
        for col in cols_to_combine:
            new_row[col] = group[col].tolist()

        # penalty for ties
        scores = new_row["scores"]
        max_val = np.max(scores)
        new_row["results"] = (1 / np.sum(scores == max_val)
                              ) if scores[0] == max_val else 0

        # new_row["results"] = 1 if np.argmax(new_row["scores"]) == 0 else 0

        # Handle all other columns - verify they're identical and take first value
        other_columns = [
            col for col in df.columns if col not in cols_to_combine]
        for col in other_columns:
            values = group[col].unique()
            if len(values) != 1:
                raise ValueError(
                    f"Column {col} has different values within group at index {current_idx}: {values}")
            new_row[col] = values[0]

        rerolled_rows.append(new_row)
        current_idx += group_size

    # Create new dataset
    rerolled_df = pd.DataFrame(rerolled_rows)
    rerolled_dataset = Dataset.from_pandas(rerolled_df)

    return rerolled_dataset


def load_bon_dataset_v2(
    dataset: str = "data/eval/rewardbench_v2.jsonl",
    custom_dialogue_formatting: bool = False,
    conv = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "text", "id"],
    local: bool = True,
):
    """
    Loads the BON candidates dataset.
    """

    # load the data. Data can be a HuggingFace dataset or a local JSONL file
    if ".jsonl" in dataset:
        raw_dataset = pd.read_json(dataset, lines=True)
        raw_dataset = Dataset.from_pandas(raw_dataset)
    elif local:
        raw_dataset = load_from_disk(dataset)
    else:
        # change split if renamed
        raw_dataset = load_dataset(dataset, split="test")

    # take column total_completions from dataset before unrolling
    total_completions = raw_dataset["total_completions"]
    num_correct = raw_dataset["num_correct"]
    logger.info(f"Total completions: {sum(total_completions)}")

    # unroll every response in chosen and rejected to a new row, all other columns are copied
    def unroll_output(idx, row):
        rows = []
        options = row["chosen"]
        options.extend(row["rejected"])

        for i, output in enumerate(options):
            new_row = row.copy()
            new_row["input"] = output
            del new_row["chosen"]
            del new_row["rejected"]
            rows.append(new_row)
        return rows

    new_dataset = []
    for idx, row in enumerate(raw_dataset):
        new_dataset.extend([r for r in unroll_output(idx, row)])

    # create huggingface dataset through pandas
    unrolled_dataset = Dataset.from_pandas(pd.DataFrame(data=new_dataset))
    # unrolled_dataset = unrolled_dataset.rename_column("index", "id")

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = unrolled_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer, "ift": True},
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = unrolled_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv, "ift": True},
                num_proc=8,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations_ift(example):
            if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
                # 多轮对话：直接追加 assistant 回复
                example["text"] = example["prompt"] + [
                    {"role": "assistant", "content": example["input"]}
                ]
            else:
                # 单轮对话：构建完整对话
                example["text"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["input"]},
                ]
            return example

        dataset = unrolled_dataset.map(
            map_conversations_ift,
            # fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    # take column subset from dataset
    subsets = dataset["subset"] if "subset" in dataset.column_names else None

    # remove column input
    all_cols = dataset.column_names
    dataset = dataset.remove_columns(
        [c for c in all_cols if c not in keep_columns])
    return dataset, subsets, total_completions, num_correct


# Helper function for scoring ties subset
def _compute_prompt_stats(samples: List[Tuple[bool, float]]) -> Tuple[bool, float | None, float | None]:
    """
    Given a list of (is_correct, score) tuples for one prompt,
    return:
        accurate ................ True if every correct answer outscores the best wrong one
        different_correct_margin  Spread between best and worst correct answers (None if <2)
        correct_incorrect_margin  Gap between worst correct and best wrong (None if N/A)
    """
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)

    # Calculate the margins with correct scores, and also the margin between correct and incorrect scores
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0

    return accurate, different_correct_margin, correct_incorrect_margin


# Processing Ties Score
def process_single_model(dataset):
    """
    Process a single-model ties evaluation dataset and return
        (dataset_with_results_column, overall_score)
    Each row in the dataset contains a list of "scores", where the first "num_correct" correspond to
        correct answers, and the rest are incorrect. The "id" field is formatted as "sample_type:prompt_id",
        where sample_type is either "ref" for reference prompts with 1 correct answer or "tied" for tied samples
        with multiple correct answers.
    Overall score is essentially 60% accuracy, 40% margin. Accuracy is broken down equally
        across ref and tied accuracy, while margin is broken down into whether the margin between
        correct answers < margin between correct and incorrect answers for tied prompts only (correctness_preferred)
        and whether this margin also holds when the margin between correct and incorrect answers is the min of the
        margin for a tied prompt and its associated reference prompt (correctness_preferred_hard).
    """
    grouped_samples: Dict[Tuple[str, int],
                          List[Tuple[bool, float]]] = defaultdict(list)

    for sample in dataset:
        # Split samples into ref and tied
        sample_type, prompt_id_str = sample["id"].split(":")
        prompt_id = int(prompt_id_str)

        # Each score position i is “correct” if i < num_correct
        for i, raw_score in enumerate(sample["scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped_samples[(sample_type, prompt_id)].append(
                (i < sample["num_correct"], score))

    # Calculate per-prompt stats
    ref_stats = {}
    tied_stats = {}

    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = _compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:  # "tied"
            tied_stats[prompt_id] = stats

    # Calculate global metrics
    # Average accuracy (element 0 of each tuple) over ref and tied samples
    ref_accuracy = np.mean([s[0]
                           for s in ref_stats.values()]) if ref_stats else 0.0
    tied_accuracy = np.mean(
        [s[0] for s in tied_stats.values()]) if tied_stats else 0.0

    # Margins: compute whether margin within correct answers < margin between correct and incorrect answers
    all_prompts = set(ref_stats) & set(tied_stats)

    # correct margin is element 1 in stats tuple, correct-incorrect margin is element 2
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])

    correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
    correctness_preferred_hard = np.mean(np.minimum(
        corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)

    # Tie-breaking term, optional, not much effect in practice
    # Normalised gap, then tanh to keep it in (‑1, 1)
    margin_scores = np.tanh(np.minimum(
        corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1)
    # if nan (divide by 0), set to 0
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))

    # Compute the overall score
    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )

    # Package results — there is less of a sense of per-prompt results for the Ties subset,
    # as overall_score is computed across the subset, so set "results" to None for clarity
    if "results" in dataset.column_names:
        dataset = dataset.remove_columns(["results"])
    results_dataset = dataset.add_column("results", [None] * len(dataset))

    return results_dataset, float(overall_score)
