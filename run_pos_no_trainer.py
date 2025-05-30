#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""
#import pacchetti
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import ClassLabel, load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import functools
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForTokenClassification
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoModel, AutoTokenizer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#defining arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="pos",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    # validating arguments from command line, if something is None gives 0
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
    
def main():
   
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner_no_trainer", args) #sending to HuggingFace anonymus data about the use of the script

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    def fix_labels(labels_list):
        fixed = []
        for labels in labels_list:
            fixed_labels = []
            for label in labels:
                if isinstance(label, float) and math.isnan(label):
                    fixed_labels.append("UNK")
                else:
                    fixed_labels.append(str(label))
            fixed.append(fixed_labels)
        return fixed
    
    if args.dataset is not None:
        #defining dataset
        df = pd.read_csv("data_v1.3/DevSet_EAGLES.txt", sep="\t",on_bad_lines="skip", quoting=3, header=None)
        df.columns = ['token', 'tag']
        #df["tag"] = df["tag"].map (map_custom_to_upos)
        sentences = []
        labels = []
        
        current_tokens = []
        current_tags = []
        #establishing column names
        for _, row in df.iterrows():
            token = row["token"]
            tag = row["tag"]
            
            current_tokens.append(token)
            current_tags.append(tag)
        
            if token in [".", "!", "?"]:
                sentences.append(current_tokens)
                labels.append(current_tags)
                current_tokens = []
                current_tags = []
        
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_tags)
        
        labels = fix_labels(labels)
        
        #defining test set for benchmark
        df_test = pd.read_csv("EVALITA07_POS_TestSet4RESULTS/TestSet_EAGLES.txt", sep="\t",on_bad_lines="skip", quoting=3, header=None)
        df_test.columns = ['token', 'tag']
        #df_test["tag"] = df_test["tag"].map(map_custom_to_upos)
        sentences_test = []
        labels_test = []
        
        current_tokens_test = []
        current_tags_test = []
        
        for _, row_test in df_test.iterrows():
            token_test = row_test["token"]
            tag_test = row_test["tag"]
            
            current_tokens_test.append(token_test)
            current_tags_test.append(tag_test)
        
            if token_test in [".", "!", "?"]:
                sentences_test.append(current_tokens_test)
                labels_test.append(current_tags_test)
                current_tokens_test = []
                current_tags_test = []
        
        if current_tokens_test:
            sentences_test.append(current_tokens_test)
            labels_test.append(current_tags_test)
        
        labels_test = fix_labels(labels_test)
        
        # Splitting 80% train, 20% validation on dataset and giving information about test set
        split_index = int(0.8 * len(sentences))
        train_data = {
            
                "tokens": sentences[:split_index],
                "labels": labels[:split_index]
            }
        val_data = {
                "tokens": sentences[split_index:],
                "labels": labels[split_index:]
            }
            
        test_data = {
                "tokens": sentences_test,
                "labels": labels_test
        }
        
        
        for i, label_list in enumerate(train_data['labels']):
            for label in label_list:
                if not isinstance(label, str):
                    print(f"Valore non stringa trovato in labels alla riga {i}: {label} (tipo {type(label)})")

        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        test_dataset = Dataset.from_dict(test_data)
        raw_datasets = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        label_list = sorted(set(tag for seq in labels for tag in seq if tag))  #removes empty spaces
        label_to_id = {label: i for i, label in enumerate(label_list)}
        
        label_list_test = sorted(set(tag_test for seq_test in labels_test for tag_test in seq_test if tag_test))  #remove empty spaces
        label_to_id_test = {label_test: i_test for i_test, label_test in enumerate(label_list_test)}
        #establishing tokenizer and model for Pos-Tagging
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        label_list = sorted(set(tag for seq in labels for tag in seq if tag))
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list),
        
            id2label=id_to_label,
            label2id=label_to_id,
        )
        
    # Preprocessing the datasets.
    # transforming dataset in numerical token in order to transfer it to the model
        padding = "max_length" if args.pad_to_max_length else False
    
    # Tokenize all texts and align the labels with them.
    
    # Defining default_label_id
    #default_label_id = 0  #neutral label

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding=padding,
            truncation=True,
            max_length=args.max_length,
        )

        labels = []
        for i, label_seq in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id.get(label_seq[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
    
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    #mapping the dataset
    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=["tokens", "labels"],
            desc="Running tokenizer on dataset",
        )
        
    
    # Converting Pandas dataframe into huggingface
    train_dataset = processed_raw_datasets["train"]
    val_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]
    
    
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # For fp8, we pad to multiple of 16.
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader,test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Metrics
    metric = evaluate.load("accuracy")
    
    #defining labels in order to compute metrics
    def get_labels(predictions, references, label_list, device, model):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()
            
        if len(y_pred.shape) > 2:  # e.g., [batch_size, seq_len, num_classes]
            y_pred = y_pred.argmax(axis=-1)    
        
        
        # Remove ignored index (special tokens)
        true_predictions = [
        # Usiamo model.config.id2label per la conversione da ID a stringa
            [model.config.id2label[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
        # Usiamo model.config.id2label per la conversione da ID a stringa
            [model.config.id2label[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]  
        return true_predictions, true_labels
    
    #computing metrics
    def compute_metrics(y_pred_nested, y_true_nested):
        y_pred_flat = [p for seq in y_pred_nested for p in seq]
        y_true_flat = [l for seq in y_true_nested for l in seq]
        

        precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="weighted", zero_division=0
    )
        acc = accuracy_score(y_true_flat, y_pred_flat)

        return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        }

    # debug example for tokenization
    for i in range(len(train_data["tokens"])):
        #example = train_data.iloc[i]
        tokens = train_data['tokens'][i]
        label = train_data['labels'][i]
        example = {"tokens": tokens, "labels": labels}
        
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    from collections import Counter
    import itertools
    
    flat_train_labels = list(itertools.chain.from_iterable(train_data["labels"]))
    Counter(flat_train_labels).most_common(20)  # top 20 POS tags

    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        #starting validation
        model.eval()
        samples_seen = 0
        all_preds=[]
        all_labels=[]
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            
            
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
                    labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels_gathered.shape[0]
            preds, refs = get_labels(predictions_gathered, labels_gathered, label_list=label_list, device=device, model=model)
            
            all_preds.extend(preds)
            all_labels.extend(refs)
            
            
              # predictions and preferences are expected to be a nested list of labels, not label_ids
        #computing evaluation metrics
        eval_metric = compute_metrics(all_preds, all_labels)
        accelerator.print(f"Risultati del dataset: epoch {epoch}:", eval_metric)
        
        

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_precision": eval_metric["precision"],
                    "eval_recall": eval_metric["recall"],
                    "eval_f1": eval_metric["f1"],
                    "eval_accuracy": eval_metric["accuracy"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    #training is done 
    #starting testing with test set
    model.eval()
    samples_seen = 0
    all_preds_test = []
    all_labels_test = []
    total_test_loss = 0.0 #inizializza la loss cumulativa

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss #calcola la loss del batch
        total_test_loss += loss.item() #aggiungila al totale
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
    
        if not args.pad_to_max_length:
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    
        predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
    
        if accelerator.num_processes > 1:
            if step == len(test_dataloader) - 1:
                predictions_gathered = predictions_gathered[: len(test_dataloader.dataset) - samples_seen]
                labels_gathered = labels_gathered[: len(test_dataloader.dataset) - samples_seen]
            else:
                samples_seen += labels_gathered.shape[0]
    
        preds, refs = get_labels(predictions_gathered, labels_gathered, label_list=label_list, device=device, model=model)
        all_preds_test.extend(preds)
        all_labels_test.extend(refs)
    
    #calcola la loss media
    avg_test_loss = total_test_loss / len(test_dataloader)
    #computing test metric
    test_metric = compute_metrics(all_preds_test, all_labels_test)
    test_metric["loss"] = avg_test_loss
    accelerator.print(f"Risultati del test set:", test_metric)

    if args.with_tracking:
        accelerator.log(
            {
                "test_precision": test_metric["precision"],
                "test_recall": test_metric["recall"],
                "test_f1": test_metric["f1"],
                "test_accuracy": test_metric["accuracy"],
                "test_loss" : avg_test_loss,
            },
            step=completed_steps,
        )
        
    if args.with_tracking:
        accelerator.end_training()
    #sending results to specific folder with argument --output_dir
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
#printing results in specific .json, first one for training and validation; second one for test set
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            if args.with_tracking:
                all_results.update({"train_loss": total_loss.item() / len(train_dataloader)})
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                # Convert all float64 & int64 type numbers to float & int for json serialization
                for key, value in all_results.items():
                    if isinstance(value, np.float64):
                        all_results[key] = float(value)
                    elif isinstance(value, np.int64):
                        all_results[key] = int(value)
                json.dump(all_results, f)
            
            all_results_test = {f"eval_{k}": v for k, v in test_metric.items()}
            with open(os.path.join(args.output_dir, "all_results_test.json"), "w") as f:
                # Convert all float64 & int64 type numbers to float & int for json serialization
                for key, value in all_results_test.items():
                    if isinstance(value, np.float64):
                        all_results_test[key] = float(value)
                    elif isinstance(value, np.int64):
                        all_results_test[key] = int(value)
                json.dump(all_results_test, f)


if __name__ == "__main__":
    main()