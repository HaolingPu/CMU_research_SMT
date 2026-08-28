# Simul-tst-COMMON

This repository provides scripts and instructions for reproducing **[Simul-tst-COMMON](https://direct.mit.edu/coli/article/doi/10.1162/COLI.a.622/136359/Rethinking-Evaluation-in-Simultaneous-Speech)**.

**Simul-tst-COMMON** is a set of test data designed for simultaneous translation settings. It focuses on maintaining the word and phrase order of the source sentence as much as possible.

The original work covers three language pairs:

* English-to-Japanese
* English-to-Chinese
* English-to-German

These language pairs represent different levels of word order similarity to English. The data was created using large language models and then quality-checked by professional interpreters.

The test sets provide an interpreter-grounded perspective on simultaneous translation. They show the ideal level of monotonicity and other sentence-level characteristics, such as syntactic simplicity and sentence length. They also reveal both the strengths and limitations of LLMs in monotonic translation.

## Workflow

### 1. Automatic Speech Recognition

Use [Whisper](https://github.com/openai/whisper) `medium.en` to transcribe the audio into English text.

```bash
whisper "${file}.wav" \
    --model medium.en \
    --output_dir ${OUTPUT_DIR}
```

### 2. Manual Correction

Check whether each transcribed text matches the corresponding MuST-C `tst-COMMON` data. If there are any ASR errors, correct them manually.

After correction, concatenate the transcriptions for each TED talk into one file.

The concatenated plain-text transcription is then split at the sentence level using punctuation marks.

```bash
python scripts/split.py --input input.txt --output output.txt
```

To obtain the cleaned source script, apply the edits.json file in the data directory to the original ASR output using the apply_edits.py. This will reproduce the manually corrected source script.

```bash
python scripts/apply_edits.py 
    --input output.txt 
    --edits data/src/edits.json    
    --output data/src/Simul-tst-COMMON.en
```

### 3. Translation with GPT

Use `create_batch.py` to create a JSONL file for batch processing.

The prompt for data creation follows [Makinae et al](https://aclanthology.org/2024.emnlp-main.1238v2.pdf).

```bash
python scripts/create_batch.py 
    --input Simul-tst-COMMON.en 
    --output_dir /path/to/
```

### 4. Batch Upload

Use `upload_batch.py` to upload the JSONL files and create OpenAI batch jobs.

```bash
python upload_batch.py 
    --lang_pairs en-{target_language} 
    --file_names tst-COMMON 
    --file_numbers 1 
    --input_dir /path/to/ 
```

### 5. Extract Batch Outputs

Use `extract_batch.py` to convert the processed batch output into the following files:

* `sep.en`
* `sep.{target_language}`
* `Simul-LLM.{target_language}`

```bash
python extract_batch.py 
    --input batch_output.jsonl 
    --output_dir /path/to/
```

After this step, you will obtain the initial LLM-generated test data, referred to as `Simul-LLM`.

### 6. Final Correction with Patch

To obtain the final `simul-tst-COMMON` data, apply the edits.json file in the data directory to `Simul-LLM` using the apply_edits.py. This will reproduce the manually corrected `Simul-tst-COMMON`.

```bash
python scripts/apply_edits.py     
    --input data/tgt/{target_language}/Simul-LLM.{target_language}     
    --edits data/{target_language}/tgt/edits.json     
    --output data/tgt/Simul-tst-COMMON.{target_language}
```

## Creating YAML Files

To create YAML files, first run **[Gentle Aligner](https://github.com/strob/gentle)** using the `.wav` audio file and the corresponding `.txt` transcript for each TED talk.  
The `.wav` and `.txt` files should belong to the same TED talk. Gentle Aligner will produce a JSON alignment file.

After collecting all JSON alignment files, run `parse_gentle.py` to convert them into YAML format.

```bash
python parse_gentle.py 
    --json_dir path/to/json 
    --txt_dir path/to/txt 
    --output path/to/Simul-tst-COMMOON.yaml
```