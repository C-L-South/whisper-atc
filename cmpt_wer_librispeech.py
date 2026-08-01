#------------------------------------------------------------------------------
# Compute WER for Fine-Tuned Whisper Models on LibriSpeech
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
# Load testing dataset
#------------------------------------------------------------------------------
!pip install datasets==3.6.0 --upgrade
from datasets import DatasetDict, load_dataset,Audio

dataset = DatasetDict()

#------------------------------------------------------------------------------
# Load LibriSpeech clean-test set 
#------------------------------------------------------------------------------
ds_clean = load_dataset("openslr/librispeech_asr", "clean", split="test")

#------------------------------------------------------------------------------
# Resample to 16k for Whisper
#------------------------------------------------------------------------------
ds_clean = ds_clean.cast_column("audio", Audio(sampling_rate=16000))
 
dataset = DatasetDict()
dataset["test_clean"] = ds_clean

print(dataset)

#------------------------------------------------------------------------------
# Install required packages
#------------------------------------------------------------------------------
!pip install -q jiwer
!pip install -q evaluate
!pip install -q openai-whisper          # required for the correct EnglishTextNormalizer

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer  # openai-whisper  
import torch
import json
import evaluate

#------------------------------------------------------------------------------
# Define the five model checkpoints to evaluate.
#------------------------------------------------------------------------------
BASE_MODEL = "cody-li/ft_wspr_sm_jt_4_8_"
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
model_paths = [BASE_MODEL + fold for fold in FOLDS]

device = "cuda" if torch.cuda.is_available() else "cpu"
metric = evaluate.load("wer")

#------------------------------------------------------------------------------
# Text normalizer
#------------------------------------------------------------------------------
import re
_base_normalizer = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    """Full Whisper English normalisation + underscore removal."""
    text = _base_normalizer(text)
    text = text.replace("_", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

#------------------------------------------------------------------------------
# NORMALIZE_PREDICTIONS flag
#------------------------------------------------------------------------------
NORMALIZE_PREDICTIONS = True  


print("Models to evaluate:")
for p in model_paths:
    print(" ", p)
print(f"\nDevice: {device}")

#------------------------------------------------------------------------------
# Evaluate all five folds
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
    ).to(device)
    model.eval()

    #--------------------------------------------------------------------------
    # Filter empty references (after normalization)
    #--------------------------------------------------------------------------
    def keep_non_empty(example):
        return len(normalize_text(example["text"])) > 0

    test_split = dataset["test_clean"].filter(keep_non_empty)
    print(f"  Test samples after filtering: {len(test_split)}")

    #--------------------------------------------------------------------------
    # Run inference
    #--------------------------------------------------------------------------
    def map_to_pred(example):
        audio = example["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).to(device)

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
            pred_ids = model.generate(
                inputs.input_features.to(device=device, dtype=model_dtype)
            )[0]

        transcription = processor.decode(pred_ids, skip_special_tokens=True)

        #----------------------------------------------------------------------
        # Reference: always normalized 
        #----------------------------------------------------------------------
        example["reference"] = normalize_text(example["text"])
        if NORMALIZE_PREDICTIONS:
            transcription = normalize_text(transcription)
        example["prediction"] = transcription
        return example

    result = test_split.map(map_to_pred)

    #--------------------------------------------------------------------------
    # Save predictions to disk
    #--------------------------------------------------------------------------
    asr_result = {"ref": list(result["reference"]), "asr": list(result["prediction"])}
    out_path = f"wer_{fold}_librispeech.json"
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

print("\n" + "="*60)
print("  WER RESULTS SUMMARY  (LibriSpeech test-clean)")
print("="*60)
print(f"  {'Model':<50}  {'WER (%)':>8}")
print("-"*60)
for fold, path in zip(FOLDS, model_paths):
    wer = fold_wer_results[fold]
    print(f"  {path:<50}  {wer:>8.2f}%")
print("-"*60)
wer_values = list(fold_wer_results.values())
print(f"  {'Mean WER':<50}  {statistics.mean(wer_values):>8.2f}%")
print(f"  {'Std Dev':<50}  {statistics.stdev(wer_values):>8.2f}%")
print("="*60)
