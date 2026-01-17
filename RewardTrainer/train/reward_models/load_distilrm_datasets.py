from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)


def build_joint_dataset(
    data_path,
    tokenizer,
    teacher_tokenizer,
    split="train",
    size=None,
    model_name="",
    max_length=512,
    score_threshold=0,  # 分数差过滤阈值
):
    try:
        ds = load_dataset("json", data_files=data_path, split="train")
        logging.info(
            f"Loaded joint dataset from {data_path} with {len(ds)} examples")
    except Exception as e:
        logging.error(f"Failed to load dataset from {data_path}: {e}")
        raise

    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
        logging.info(f"Reduced dataset to {len(ds)} examples")

    # --------------------------
    # Convert to chat format
    # --------------------------
    def convert_to_chat_format(example):
        prompt = example.get("prompt", "").strip()
        chosen = example.get("chosen", "").strip() or " "
        rejected = example.get("rejected", "").strip() or " "
        teacher_response = example.get("teacher_response", "").strip() or " "

        if not prompt:
            raise ValueError(f"Empty prompt in example: {example}")

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]

        result = {
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "teacher_response": teacher_response,
            "prompt": prompt,
            "chosen_text": chosen,
            "rejected_text": rejected,
        }

        if "chosen_score" in example:
            result["chosen_score"] = example["chosen_score"]
        if "rejected_score" in example:
            result["rejected_score"] = example["rejected_score"]

        return result

    # --------------------------
    # Tokenization
    # --------------------------
    def formatting_func(example):
        try:
            example = convert_to_chat_format(example)

            chosen = example["chosen_text"]
            rejected = example["rejected_text"]

            # 默认标记有效
            valid = True

            # Tokenizer kwargs
            kwargs = {
                "padding": "max_length",
                "truncation": True,
                "max_length": max_length,
                "return_tensors": "pt",
            }

            # --- chosen & rejected tokenization ---
            prompt_plus_chosen_response = tokenizer.apply_chat_template(
                example["chosen"], tokenize=False
            )
            prompt_plus_rejected_response = tokenizer.apply_chat_template(
                example["rejected"], tokenize=False
            )
            tokens_chosen = tokenizer.encode_plus(
                prompt_plus_chosen_response, **kwargs)
            tokens_rejected = tokenizer.encode_plus(
                prompt_plus_rejected_response, **kwargs)

            # --- prefix masking ---
            prompt = example["chosen"][:-1]
            prompt_template = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            tokens_prompt = tokenizer.encode_plus(
                prompt_template, add_special_tokens=False,
                truncation=True, return_tensors="pt"
            )["input_ids"][0]

            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_chosen[: len(tokens_prompt)] = -100
            label_rejected[: len(tokens_prompt)] = -100

            # --- SFT student data (teacher_response) ---
            student_teacher_messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["teacher_response"]},
            ]
            student_teacher_text = tokenizer.apply_chat_template(
                student_teacher_messages, tokenize=False
            )
            student_teacher_tokens = tokenizer.encode_plus(
                student_teacher_text, **kwargs
            )
            student_teacher_labels = student_teacher_tokens["input_ids"][0].clone(
            )
            student_teacher_labels[: len(tokens_prompt)] = -100

            # --- teacher tokenizer ---
            if teacher_tokenizer:
                teacher_messages = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant",
                        "content": example["teacher_response"]},
                ]
                teacher_text = teacher_tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False
                )
                teacher_tokens = teacher_tokenizer(teacher_text, **kwargs)

                teacher_user_only_text = teacher_tokenizer.apply_chat_template(
                    [{"role": "user", "content": example["chosen"][0]["content"]}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                teacher_user_only_ids = teacher_tokenizer.encode_plus(
                    teacher_user_only_text, add_special_tokens=False,
                    truncation=True, max_length=max_length,
                    return_tensors="pt"
                )["input_ids"][0]
                teacher_prefix_len = len(teacher_user_only_ids)
            else:
                teacher_tokens = tokens_chosen
                teacher_prefix_len = len(tokens_prompt)

            student_prefix_len = len(tokens_prompt)

            result = {
                # RM inputs
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected,

                # SFT student data
                "student_teacher_input_ids": student_teacher_tokens["input_ids"][0],
                "student_teacher_attention_mask": student_teacher_tokens["attention_mask"][0],
                "student_teacher_labels": student_teacher_labels,

                # teacher tokenizer outputs
                "teacher_input_ids": teacher_tokens["input_ids"][0],
                "teacher_attention_mask": teacher_tokens["attention_mask"][0],
                "teacher_prefix_len": teacher_prefix_len,
                "student_prefix_len": student_prefix_len,

                "__valid__": valid,
            }

            # keep scores
            if "chosen_score" in example:
                result["chosen_score"] = example["chosen_score"]
            if "rejected_score" in example:
                result["rejected_score"] = example["rejected_score"]

            return result

        except Exception as e:
            logging.error(f"Error processing example {example}: {e}")
            raise

    # --------------------------
    # Step 3: tokenization
    # --------------------------
    ds = ds.map(formatting_func, batched=False, num_proc=16)

    # --------------------------
    # Step 4: filter only valid items
    # --------------------------
    ds = ds.filter(lambda x: x["__valid__"] is True)

    logging.info(f"After filtering valid: {len(ds)} examples")

    # --------------------------
    # Step 5: validation checks
    # --------------------------
    def validate_tokens(example):
        rm_keys = [
            ("input_ids_chosen", "attention_mask_chosen"),
            ("input_ids_rejected", "attention_mask_rejected"),
        ]
        for ids_key, mask_key in rm_keys:
            if len(example[ids_key]) != max_length or len(example[mask_key]) != max_length:
                return False
            if sum(example[mask_key]) <= 1:
                return False

        sft_keys = [("student_teacher_input_ids",
                     "student_teacher_attention_mask")]
        for ids_key, mask_key in sft_keys:
            if len(example[ids_key]) != max_length or len(example[mask_key]) != max_length:
                return False

        teacher_keys = [("teacher_input_ids", "teacher_attention_mask")]
        for ids_key, mask_key in teacher_keys:
            if len(example[ids_key]) != max_length or len(example[mask_key]) != max_length:
                return False

        return True

    ds = ds.filter(validate_tokens)
    logging.info(f"Joint dataset after validation: {len(ds)} examples")

    # --------------------------
    # Step 6: remove unused columns
    # --------------------------
    remove_cols = [c for c in ds.column_names if c.startswith("__")]
    ds = ds.remove_columns(remove_cols)

    ds.set_format(type="torch")
    return ds


def load_joint_train_eval_dataset(
    data_path,
    tokenizer,
    teacher_tokenizer,
    size=None,
    model_name="",
    max_length=512,
):
    dataset = build_joint_dataset(
        data_path,
        tokenizer,
        teacher_tokenizer,
        split="train",
        size=size,
        model_name=model_name,
        max_length=max_length,
    )
    dataset_split = dataset.train_test_split(test_size=0.01)
    train_dataset, eval_dataset = dataset, dataset_split["test"]

    logging.info(
        f"Joint train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}"
    )
    return train_dataset, eval_dataset
