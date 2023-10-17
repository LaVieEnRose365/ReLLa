import os
import sys
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=256)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--use_lora", type=int, default=0)
parser.add_argument("--dataset", type=str, default='BookCrossing')

# Here are args of prompt
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--temp_type", type=str, default="simple")

args = parser.parse_args()

# assert args.temp_type in ["simple", "sequential"]
assert args.dataset in ['ml-1m', 'BookCrossing', 'GoodReads']
data_path = f"./data/{args.dataset}/proc_data/data"

print("\n")
print('*'*50)
print(args)
print('*'*50)
print("\n")

transformers.set_seed(args.seed)

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"

MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 5  # we don't always need 3 tbh
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size #2000
USE_8bit = True

if USE_8bit is True:
    warnings.warn("If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2")
        
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
    
DATA_PATH = {
    "test": '/'.join([data_path, f"test/test_{args.K}_{args.temp_type}.json"])
}


OUTPUT_DIR = args.output_path #"lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(args.model_path)
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=USE_8bit,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

if USE_8bit is True:
    model = prepare_model_for_int8_training(model)

if args.use_lora:
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#tokenizer.padding_side = "left"  # Allow batched inference


data = load_dataset("json", data_files=DATA_PATH)
print("Data loaded.")

if args.use_lora:
    print("Load lora weights")
    adapters_weights = torch.load(os.path.join(args.resume_from_checkpoint, "pytorch_model.bin"))
    set_peft_model_state_dict(model, adapters_weights)
    print("lora load results")

def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "\
            f"USER: {data_point['input']} ASSISTANT: "
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    ) - 1  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        # padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


test_data = data['test'].map(generate_and_tokenize_prompt)
print("Data processed.")


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


trainer = transformers.Trainer(
    model=model,
    train_dataset=test_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=False,
        logging_steps=1,
        evaluation_strategy="epoch" if VAL_SET_SIZE > 0 else "no",
        save_strategy="epoch",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=30,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding='longest'
    ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

print("Evaluate on the test set...")
print(trainer.evaluate(eval_dataset=test_data))


