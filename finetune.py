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
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)

# Here are args of prompt
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--train_type", type=str, default="simple")
parser.add_argument("--test_type", type=str, default="simple")

args = parser.parse_args()

assert args.train_type in ["simple", "sequential", "mixed", "high"]
assert args.test_type in ["simple", "sequential", "high"]
assert args.dataset in ["ml-1m", "BookCrossing", "ml-25m"]

data_path = f"./data/{args.dataset}/proc_data/data"

# Fit for single card V100, increasing bs if GPU allows is OK.
if args.K <= 15:
    args.per_device_eval_batch_size = 8
elif args.K <= 40:
    args.per_device_eval_batch_size = 4
else:
    args.per_device_eval_batch_size = 2


print('*'*100)
print(args)
print('*'*100)

transformers.set_seed(args.seed)

print(f"Shot: {args.train_size}")
if args.train_type == "mixed":
    args.train_size *= 2
print(f"Samples used: {args.train_size}")

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"


MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size #2000
USE_8bit = True
OUTPUT_DIR = args.output_path

if USE_8bit is True:
    warnings.warn("If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2")
        
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]



DATA_PATH = {
    "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}_sampled.json"]), 
    "test": '/'.join([data_path, f"test/test_{args.K}_{args.test_type}.json"])
}


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
    print("Lora used.")

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#tokenizer.padding_side = "left"  # Allow batched inference


data = load_dataset("json", data_files=DATA_PATH)
data["train"] = data["train"].select(range(args.train_size))
print("Data loaded.")



now_max_steps = max((len(data["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)
if args.resume_from_checkpoint:
    if args.lora_remote_checkpoint is not None:
        snapshot_download(repo_id=args.lora_remote_checkpoint, allow_patterns=["*.pt", "*.bin", "*.json"], local_dir=args.resume_from_checkpoint)
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    
    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    
    if os.path.exists(train_args_path):
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps


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
    )["input_ids"][:-1]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


test_data = data['test'].map(generate_and_tokenize_prompt)
train_data = data["train"].map(generate_and_tokenize_prompt)
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
    train_dataset=train_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=30,
        load_best_model_at_end=True,
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
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)
model.config.use_cache = False

if args.use_lora:
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)
    

print("Start training...")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)

