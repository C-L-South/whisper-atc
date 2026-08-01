#------------------------------------------------------------------------------
# Compute WER for Fine-Tuned Whisper Models on Jacktol ATC Test Dataset
#
# Instructions:
# Copy the entire code and paste it into a Google Colab notebook. Then, set 
# the model name to evaluate and run the notebook end-to-end.
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
# Define token and setup login
#------------------------------------------------------------------------------
from huggingface_hub import login

try:
    #--------------------------------------------------------------------------
    # read from Colab Secrets  
    #--------------------------------------------------------------------------
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    #--------------------------------------------------------------------------
    # prompt interactively
    #--------------------------------------------------------------------------
    import getpass
    hf_token = getpass.getpass("Paste your Hugging Face token: ")

login(token=hf_token)
print("Logged in to Hugging Face.")

#------------------------------------------------------------------------------
# Load the Jacktol ATC dataset (both "train" and "test" splits).
#------------------------------------------------------------------------------
!pip install -q "datasets==3.6.0" --upgrade
!pip install -q openai-whisper           

from datasets import load_dataset

dataset = load_dataset("jacktol/atc-dataset")   
print(dataset)

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import json
!pip install evaluate
import evaluate
from whisper.normalizers import EnglishTextNormalizer  

#------------------------------------------------------------------------------
# NORMALIZE_PREDICTIONS flag
#------------------------------------------------------------------------------
NORMALIZE_PREDICTIONS = False  # set to True when evaluating the baseline model

#------------------------------------------------------------------------------
# Base model path and fold suffixes
#------------------------------------------------------------------------------
BASE_MODEL = "cody-li/ft_wspr_sm_jt_4_8_"
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
model_paths = [BASE_MODEL + fold for fold in FOLDS]

print("Models to evaluate:")
for p in model_paths:
    print(" ", p)

#------------------------------------------------------------------------------
# Filter the test set once to remove any samples that are too short or have 
# empty transcripts. This is done once here to avoid repeating the filtering 
# logic for each fold. set Duration_sec >= 0.0 to use all samples, 
# including very short ones. 
#------------------------------------------------------------------------------
def is_valid(example):
    audio        = example["audio"]
    duration_sec = len(audio["array"]) / audio["sampling_rate"]
    transcript   = example["text"].strip()
    return duration_sec >= 0.0 and len(transcript) > 0

test_set_clean = dataset["test"].filter(is_valid)
#test_set_clean = dataset["test"]

print(f"Test set size after filtering: {len(test_set_clean)}")

print(test_set_clean)

#------------------------------------------------------------------------------
# Define normalisation function for both reference and prediction text.  
#------------------------------------------------------------------------------
import re
_base_normalizer = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    """Full Whisper English normalisation + underscore removal."""
    text = _base_normalizer(text)
    text = text.replace("_", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

!pip install jiwer
metric = evaluate.load("wer")

#------------------------------------------------------------------------------
# Evaluate all folds
#------------------------------------------------------------------------------
fold_wer_results = {}  

for fold, model_path in zip(FOLDS, model_paths):
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_path}")
    print(f"{'='*70}")

    #--------------------------------------------------------------------------
    # Load model and processor for this fold  
    #--------------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to("cuda")
    model.eval()

    #--------------------------------------------------------------------------
    # Inference  
    #--------------------------------------------------------------------------
    def map_to_pred(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features

        #----------------------------------------------------------------------
        # Normalize the reference text 
        #----------------------------------------------------------------------
        batch["reference"] = normalize_text(batch["text"])

        is_multilingual = getattr(model.generation_config, "is_multilingual", None)
        if is_multilingual is None:
            is_multilingual = getattr(processor.tokenizer, "is_multilingual", None)
        if is_multilingual is None:
            is_multilingual = "<|en|>" in getattr(processor.tokenizer, "additional_special_tokens", [])

        if is_multilingual:
            dec_prompt = processor.get_decoder_prompt_ids(language="en", task="transcribe")
            model.generation_config.forced_decoder_ids = dec_prompt
        else:
            model.generation_config.forced_decoder_ids = None

        with torch.no_grad():
            model_dtype = next(model.parameters()).dtype
            predicted_ids = model.generate(
                input_features.to(device="cuda", dtype=model_dtype)
            )[0]

        transcription = processor.decode(predicted_ids, skip_special_tokens=True)

        if NORMALIZE_PREDICTIONS:
            transcription = normalize_text(transcription)
        batch["prediction"] = transcription
        return batch

    result = test_set_clean.map(map_to_pred)

    #--------------------------------------------------------------------------
    # Save predictions to disk  
    #--------------------------------------------------------------------------
    asr_result = {"ref": list(result["reference"]), "asr": list(result["prediction"])}
    out_path   = f"wer_{fold}.json"
    with open(out_path, "w") as fout:
        json.dump(asr_result, fout, indent=4)
    print(f"  Predictions saved to {out_path}")

    #--------------------------------------------------------------------------
    # Compute WER  
    #--------------------------------------------------------------------------
    wer_pct = 100 * metric.compute(
        references=asr_result["ref"],
        predictions=asr_result["asr"],
    )
    fold_wer_results[fold] = wer_pct
    print(f"  WER ({fold}): {wer_pct:.2f}%")

    #--------------------------------------------------------------------------
    # Free GPU memory before next fold  
    #--------------------------------------------------------------------------
    del model, processor
    torch.cuda.empty_cache()

#------------------------------------------------------------------------------
# WER summary across all folds
#------------------------------------------------------------------------------
import statistics

print("\n" + "="*50)
print("  WER RESULTS SUMMARY")
print("="*50)
print(f"  {'Model':<45}  {'WER (%)':>8}")
print("-"*50)
for fold, path in zip(FOLDS, model_paths):
    wer = fold_wer_results[fold]
    print(f"  {path:<45}  {wer:>8.2f}%")
print("-"*50)
wer_values = list(fold_wer_results.values())
print(f"  {'Mean WER':<45}  {statistics.mean(wer_values):>8.2f}%")
print(f"  {'Std Dev':<45}  {statistics.stdev(wer_values):>8.2f}%")
print("="*50)
