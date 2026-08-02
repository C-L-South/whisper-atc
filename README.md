The *.py files in this folder are used to reproduce the results from the paper "Study of LoRA Rank Selection for Fine-Tuning Whisper on Air Traffic Control Communications" by Cody Li and Jianhua Liu.

These scripts were exported directly from Google Colab notebooks. To run them:

Open a new Google Colab notebook.
Copy the entire contents of the script and paste it into the notebook.
Run the notebook end-to-end.

Scripts:

finetune_jt_5f_lora_unsloth.py - Fine-tunes Whisper (small.en) on the ATC dataset using Unsloth LoRA with 5-fold cross-validation.

cmpt_wer_librispeech.py - Computes WER (Word Error Rate) on the LibriSpeech test dataset for a given Whisper model.

cmpt_wer_jt.py - Computes WER on the ATC test dataset for a given Whisper model.

Fine-Tuned Models:

All fine-tuned Whisper samll English models are hosted on Hugging Face at https://huggingface.co/cody-li

Model naming convention:

cody-li/ft_wspr_sm_jt_{rank}_{alpha}_fold{ID}

Where:
rank = 4, 8, 16, 32, 64, 128, 256
alpha = 2 x rank
ID = 1, 2, 3, 4, 5 (fold number)

This results in a total of 35 models.
