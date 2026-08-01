#------------------------------------------------------------------------------
# Fine-tune Whisper (small.en) on the Jacktol ATC Dataset Unsloth LoRA
# with 5-Fold Cross-Validation 
# 
# Instructions:
# Copy the entire code and paste it into a Google Colab notebook. Then, set 
# the parameters in Section 1 below and run the notebook end-to-end.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Disclaimer:
# This code is shared for the purpose of promoting reproducibility and open 
# research. No copyright is claimed over this code, and it may be freely used, 
# modified, and distributed without restriction. The code is provided "as is," 
# with no warranty of any kind. The author(s) assume no responsibility or 
# liability for any errors, omissions, or consequences arising from its use, 
# including but not limited to incorrect results, data loss, or damages of any 
# kind. Users are solely responsible for verifying the correctness of results
# obtained through use of this code.
#
# Cody Li, 2026-07-02
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 1. CONFIGURATION
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LoRA hyperparameters
#------------------------------------------------------------------------------
LORA_RANK  = 64    # LoRA rank (r)
LORA_ALPHA = 128   # LoRA alpha (scaling factor 2*r)

#------------------------------------------------------------------------------
# Base model and dataset
#------------------------------------------------------------------------------
base_model_name = "openai/whisper-small.en"
dataset_name    = "jacktol/atc-dataset"

#------------------------------------------------------------------------------
# Quick test-run mode. When TEST_RUN_FLAG = True, only a small fraction
# (TEST_RUN_FRACTION) of the train split is used for the ENTIRE pipeline below.
# This lets the whole notebook (including the Hugging Face push) be tested 
# end-to-end in a few minutes instead of a full run, so you can confirm 
# everything works before switching to TEST_RUN_FLAG = False
#------------------------------------------------------------------------------
TEST_RUN_FLAG     = False  # set to False for the full training run
TEST_RUN_FRACTION = 0.02   # fraction of the train split used when TEST_RUN_FLAG=True

#------------------------------------------------------------------------------
# Cross-validation settings. CV_RANDOM_SEED fixes BOTH the dataset shuffle
# and the KFold split.
#------------------------------------------------------------------------------
N_FOLDS        = 5
CV_RANDOM_SEED = 42   

#------------------------------------------------------------------------------
# Which fold to run in THIS execution.
#------------------------------------------------------------------------------
FOLD_ID = 1  # Valid values: 1, 2, 3, 4, 5  

assert FOLD_ID in range(1, N_FOLDS + 1), f"FOLD_ID must be between 1 and {N_FOLDS}, got {FOLD_ID}"

#------------------------------------------------------------------------------
# Training duration and early stopping.
#------------------------------------------------------------------------------
TRAIN_EPOCH_NUM         = 20      
EARLY_STOPPING_PATIENCE = 3    # stop after this many evals (epochs) with no WER improvement

#------------------------------------------------------------------------------
# Optimizer and LR schedule settings. 
#------------------------------------------------------------------------------
LEARNING_RATE     = 2e-4
WARMUP_STEPS      = 500    
WEIGHT_DECAY      = 0.01
LR_SCHEDULER_TYPE = "cosine"
TRAINER_SEED      = 3407

#------------------------------------------------------------------------------
# Hugging Face upload settings and local output layout.
#------------------------------------------------------------------------------
hf_upload_org     = "cody-li"                            
output_root       = "whisper_jt_cv_train"                # local working directory for all fold outputs
fold_results_dir  = f"{output_root}/fold_results"        # one JSON file per completed fold, read by Section 12

import os
os.makedirs(output_root, exist_ok=True)
os.makedirs(fold_results_dir, exist_ok=True)

def model_name_for_fold(fold_number: int) -> str:
    #--------------------------------------------------------------------------
    # Required naming convention: ft_wspr_sm_jt_<rank>_<alpha>_fold<fold_number>
    #--------------------------------------------------------------------------
    return f"ft_wspr_sm_jt_{LORA_RANK}_{LORA_ALPHA}_fold{fold_number}"

print("Example fold-1 model name:", model_name_for_fold(1))
print(f"This execution will train FOLD_ID = {FOLD_ID}")

#------------------------------------------------------------------------------
# 2. Install dependencies
#------------------------------------------------------------------------------
!pip install -q "unsloth==2026.6.7" "unsloth_zoo==2026.6.7"
!pip install -q peft
!pip install -q "datasets>=3.4.1,<4.0.0"
!pip install -q jiwer
!pip install -q evaluate
!pip install -q scikit-learn
!pip install -q openai-whisper          

#------------------------------------------------------------------------------
# 3. Hugging Face login
#------------------------------------------------------------------------------
from huggingface_hub import login

try:
    #--------------------------------------------------------------------------
    # Read from google Colab Secrets 
    #--------------------------------------------------------------------------
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    #--------------------------------------------------------------------------
    # Prompt interactively 
    #--------------------------------------------------------------------------
    import getpass
    hf_token = getpass.getpass("Paste your Hugging Face token: ")

login(token=hf_token)
print("Logged in to Hugging Face.")

#------------------------------------------------------------------------------
# 4. Load the dataset
#------------------------------------------------------------------------------
from datasets import load_dataset

atc0 = load_dataset(dataset_name, token=hf_token, trust_remote_code=True)
print(atc0)

#------------------------------------------------------------------------------
# Shuffle once with a fixed seed
#------------------------------------------------------------------------------
full_train_dataset = atc0["train"].shuffle(seed=CV_RANDOM_SEED)

#------------------------------------------------------------------------------
# TEST_RUN_FLAG support 
#------------------------------------------------------------------------------
if TEST_RUN_FLAG:
    n_test_run = max(N_FOLDS, int(len(full_train_dataset) * TEST_RUN_FRACTION))
    full_train_dataset = full_train_dataset.select(range(n_test_run))
    print(f"TEST_RUN_FLAG=True -> using {n_test_run} / {len(atc0['train'])} examples "
          f"(~{TEST_RUN_FRACTION:.0%}) for this run.")

print(full_train_dataset)

#------------------------------------------------------------------------------
# 4b. Audio duration filter
#
# Exclude training samples whose audio is shorter than a predefined value.
# Disable this logic by setting the threshold to 0.0 (no filtering). 
#------------------------------------------------------------------------------
n_before_duration_filter = len(full_train_dataset)

def is_audio_long_enough(example):
    audio        = example["audio"]
    duration_sec = len(audio["array"]) / audio["sampling_rate"]
    return duration_sec >= 0.0

full_train_dataset = full_train_dataset.filter(is_audio_long_enough)
n_after_duration_filter = len(full_train_dataset)

print(f"Audio duration filter:")
print(f"  Before : {n_before_duration_filter} samples")
print(f"  After  : {n_after_duration_filter} samples")
print(f"  Removed: {n_before_duration_filter - n_after_duration_filter} samples")

#------------------------------------------------------------------------------
# 5. Text normalizer
#
# Uses openai-whisper's EnglishTextNormalizer
#------------------------------------------------------------------------------
import re
from whisper.normalizers import EnglishTextNormalizer as _WhisperENNorm

_base_normalizer = _WhisperENNorm()

def english_text_normalizer(text: str) -> str:
    """Full Whisper English normalisation + underscore removal."""
    text = _base_normalizer(text)
    text = text.replace("_", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

#------------------------------------------------------------------------------
# 6. Filter empty transcripts
#------------------------------------------------------------------------------
def is_transcript_empty(transcript):
    return len(transcript) > 0

full_train_dataset = full_train_dataset.filter(is_transcript_empty, input_columns=["text"])
print(full_train_dataset)

#------------------------------------------------------------------------------
# 7. Processor and feature extraction
#------------------------------------------------------------------------------
from transformers import WhisperProcessor

processor         = WhisperProcessor.from_pretrained(base_model_name)
feature_extractor = processor.feature_extractor
tokenizer         = processor.tokenizer

#------------------------------------------------------------------------------
# Feature extraction
#------------------------------------------------------------------------------
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(english_text_normalizer(batch["text"])).input_ids
    return batch

processed_dataset = full_train_dataset.map(
    prepare_dataset,
    remove_columns=full_train_dataset.column_names,
    num_proc=1,
)
print(processed_dataset)

#------------------------------------------------------------------------------
# 8. Data collator
#
# Pad zeros for the feature input and set corresponding label padding positions
#  to -100 so they're ignored by the loss.
#------------------------------------------------------------------------------
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        #----------------------------------------------------------------------
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods first treat the audio inputs by
        # simply returning torch tensors
        #----------------------------------------------------------------------
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch          = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        #----------------------------------------------------------------------
        # get the tokenized label sequences
        #----------------------------------------------------------------------
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        #----------------------------------------------------------------------
        # pad the labels to max length
        #----------------------------------------------------------------------
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        #----------------------------------------------------------------------
        # replace padding with -100 to ignore loss correctly
        #----------------------------------------------------------------------
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        #----------------------------------------------------------------------
        # if bos token is appended in previous tokenization step, cut bos token
        # here as it's appended later anyway
        #----------------------------------------------------------------------
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

#------------------------------------------------------------------------------
# 9. WER metric 
#------------------------------------------------------------------------------
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    #--------------------------------------------------------------------------
    # replace -100 with the pad_token_id (restore real label ids before
    #  decoding)
    #--------------------------------------------------------------------------
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#------------------------------------------------------------------------------
# 10. 5-fold split (always computed for all 5 folds)
#------------------------------------------------------------------------------
from sklearn.model_selection import KFold
import numpy as np

kfold = KFold(n_splits=N_FOLDS, shuffle=False)

#------------------------------------------------------------------------------
# Computed for ALL N_FOLDS folds every run (independent of FOLD_ID) so the
# partitioning is identical across separate executions.
#------------------------------------------------------------------------------
fold_indices = list(kfold.split(np.arange(len(processed_dataset))))
for i, (train_idx, val_idx) in enumerate(fold_indices, start=1):
    marker = "  <-- FOLD_ID selected for training this run" if i == FOLD_ID else ""
    print(f"Fold {i}: {len(train_idx)} train examples, {len(val_idx)} validation examples{marker}")

#------------------------------------------------------------------------------
# 11. Train the selected fold (`FOLD_ID`) with early stopping
#------------------------------------------------------------------------------
from unsloth import FastModel
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers.utils.notebook import NotebookProgressCallback
from peft import PeftModel
import gc, json, torch

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def get_best_epoch(trainer):
    #--------------------------------------------------------------------------
    # Epoch at which the lowest validation WER was actually logged.
    #--------------------------------------------------------------------------
    eval_logs = [log for log in trainer.state.log_history if "eval_wer" in log]
    return min(eval_logs, key=lambda log: log["eval_wer"])["epoch"]

#------------------------------------------------------------------------------
# Select the fold requested via FOLD_ID 
#------------------------------------------------------------------------------
fold_number        = FOLD_ID
train_idx, val_idx = fold_indices[FOLD_ID - 1]

model_name       = model_name_for_fold(fold_number)
fold_output_dir  = f"{output_root}/{model_name}"
adapter_dir      = f"{fold_output_dir}/adapter"
merged_local_dir = f"{fold_output_dir}/merged"
hub_repo_id      = f"{hf_upload_org}/{model_name}"

print("\n" + "=" * 80)
print(f"FOLD {fold_number}/{N_FOLDS}  ->  {model_name}")
print("=" * 80)

#------------------------------------------------------------------------------
# This fold's non-overlapping train / validation splits.
#------------------------------------------------------------------------------
train_fold = processed_dataset.select(train_idx)
val_fold   = processed_dataset.select(val_idx)

#------------------------------------------------------------------------------
# Load a copy of the base Whisper model for this fold via Unsloth's FastModel
#------------------------------------------------------------------------------
torch.cuda.empty_cache()
base_model, _ = FastModel.from_pretrained(
    model_name   = base_model_name,
    dtype        = None,    # auto-detect fp16/bf16
    load_in_4bit = False,   # 16-bit LoRA, matching the baseline (no QLoRA)
    auto_model   = WhisperForConditionalGeneration,
)

model = FastModel.get_peft_model(
    base_model,
    r                          = LORA_RANK,                # Configurable
    target_modules             = [                          
                                   "q_proj",                
                                   "k_proj",                
                                   "v_proj",                
                                   "out_proj",              
                                 ],
    lora_alpha                 = LORA_ALPHA,               # Configurable
    lora_dropout               = 0.05,                     
    bias                       = "none",                   
    use_gradient_checkpointing = "unsloth",                # Unsloth's memory-efficient checkpointing
    random_state               = CV_RANDOM_SEED,
    use_rslora                 = False,                   
    loftq_config               = None,
    task_type                  = None,                     # must be None for Whisper / seq2seq audio models
)
model.print_trainable_parameters()

#------------------------------------------------------------------------------
# Parameter accounting
#------------------------------------------------------------------------------
total_model_params  = sum(p.numel() for p in model.parameters())
lora_params         = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
base_model_params   = total_model_params - lora_params
lora_pct_of_base    = 100 * lora_params / base_model_params

print("\n" + "-" * 80)
print("PARAMETER COUNTS")
print("-" * 80)
print(f"Total parameters in base model:         {base_model_params:,}")
print(f"Newly added LoRA parameters:             {lora_params:,}")
print(f"LoRA parameters as % of base model:      {lora_pct_of_base:.4f}%")
print("-" * 80)

model.generation_config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False  

#------------------------------------------------------------------------------
# Warmup steps 
#------------------------------------------------------------------------------
train_batch_size = 8
grad_accum_steps = 2    
steps_per_epoch  = max(1, len(train_fold) // (train_batch_size * grad_accum_steps))
warmup_steps     = WARMUP_STEPS   

#------------------------------------------------------------------------------
# Training configuration. 
#------------------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=fold_output_dir,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    learning_rate=LEARNING_RATE,                  
    lr_scheduler_type=LR_SCHEDULER_TYPE,            
    warmup_steps=warmup_steps,                      
    weight_decay=WEIGHT_DECAY,                      
    optim="adamw_torch",                          
    seed=TRAINER_SEED,                              
    report_to="none",
    num_train_epochs=TRAIN_EPOCH_NUM,               
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,                             
    load_best_model_at_end=True,                     
    metric_for_best_model="wer",                    
    greater_is_better=False,                        # lower WER is better
    fp16=True,
    per_device_eval_batch_size=8,        
    generation_max_length=256,                                                            
    logging_steps=50,                                                
    remove_unused_columns=False,   
    label_names=["labels"],        
    predict_with_generate=True,
    push_to_hub=False,                              # push the model manually below, not raw checkpoints
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_fold,
    eval_dataset=val_fold,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,  
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],  
)

#------------------------------------------------------------------------------
# Train this fold's LoRA adapter. May stop before TRAIN_EPOCH_NUM epochs
#------------------------------------------------------------------------------
import time
fold_train_start = time.time()
trainer.train()
fold_train_end      = time.time()
fold_train_duration = fold_train_end - fold_train_start     # seconds
total_epochs_run    = trainer.state.epoch                    
best_epoch          = get_best_epoch(trainer)                

fold_h = int(fold_train_duration // 3600)
fold_m = int((fold_train_duration % 3600) // 60)
fold_s = int(fold_train_duration % 60)
print(f"Fold {fold_number} training time: {fold_h:02d}h {fold_m:02d}m {fold_s:02d}s "
      f"({fold_train_duration:.1f} s total)")

try:
    trainer.remove_callback(NotebookProgressCallback)
except Exception:
    pass

#------------------------------------------------------------------------------
# Final validation WER for this fold
#------------------------------------------------------------------------------
eval_metrics = trainer.evaluate()
fold_wer = eval_metrics["eval_wer"]
print(f"Fold {fold_number} validation WER: {fold_wer:.2f}  "
      f"(best checkpoint at epoch {best_epoch:.1f}; "
      f"training continued to epoch {total_epochs_run:.1f} before stopping, "
      f"cap={TRAIN_EPOCH_NUM})")

#------------------------------------------------------------------------------
# Save this fold's result to disk
#------------------------------------------------------------------------------
fold_result = {
    "fold": fold_number,
    "wer": fold_wer,
    "best_epoch": best_epoch,
    "total_epochs_run": total_epochs_run,
    "training_duration_sec": round(fold_train_duration, 1),
    "model_name": model_name,
}
fold_result_path = f"{fold_results_dir}/fold{fold_number}_result.json"
with open(fold_result_path, "w") as f:
    json.dump(fold_result, f, indent=2)
print(f"Saved fold result to {fold_result_path}")

#------------------------------------------------------------------------------
# Save the LoRA adapter locally
#------------------------------------------------------------------------------
model.save_pretrained(adapter_dir)

#------------------------------------------------------------------------------
# Free this fold's training model and trainer before loading a clean base model
# to merge into.
#------------------------------------------------------------------------------
del trainer, model, base_model
free_gpu_memory()

#------------------------------------------------------------------------------
# Load a clean base model, apply the saved (best) adapter, merge
# weights.
#------------------------------------------------------------------------------
clean_base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to("cuda")
merged_model = PeftModel.from_pretrained(clean_base_model, adapter_dir)
merged_model = merged_model.merge_and_unload()

#------------------------------------------------------------------------------
# Save the merged model and processor locally.
#------------------------------------------------------------------------------
merged_model.save_pretrained(merged_local_dir)
processor.save_pretrained(merged_local_dir)

#------------------------------------------------------------------------------
# Push the merged model to the Hugging Face Hub
#------------------------------------------------------------------------------
merged_model.push_to_hub(hub_repo_id, token=hf_token)
processor.push_to_hub(hub_repo_id, token=hf_token)
print(f"Pushed merged model to: https://huggingface.co/{hub_repo_id}")

del clean_base_model, merged_model
free_gpu_memory()

print(f"\nFold {fold_number} complete.")
