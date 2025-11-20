# Granary → MFA → LLM Segmentation Streaming Dataset Pipeline

This repository contains a full pipeline that converts raw audio-text data
(from Parquet files and LLM-generated segmentation JSONs) into 
high-quality **bilingual streaming translation segments**, suitable for
simultaneous translation models or low-latency MT research.

The pipeline has two layers:

1. A **single-utterance demo** (for debugging and validation)
2. A **full batch processor** (mult.py) — the production pipeline

This README documents the final **batch pipeline**.


------------------------------------------------------------
1. Data Sources
------------------------------------------------------------

The pipeline consumes three types of inputs:

1. **ASR-only Parquet files**
   - Contain raw audio bytes + transcripts per utterance

2. **MFA forced alignment output**
   - One TextGrid file per utterance
   - Must contain at least a "words" tier

3. **LLM segmentation JSONs**
   - Low-/medium-/high-latency segmentation
   - English and Chinese chunk sequences:
        {
          "low_latency": {
            "English": [...],
            "Chinese": [...]
          },
          "medium_latency": {...},
          "high_latency": {...}
        }


------------------------------------------------------------
2. MFA Corpus Generation (Optional)
------------------------------------------------------------

Before running MFA, you may generate a corpus:

    mfa_corpus/
        utt_0.wav
        utt_0.lab
        utt_1.wav
        ...

This step uses:
- raw audio bytes decoded using soundfile
- corresponding transcript text written to .lab

(You only generate corpus once; mult.py does NOT repeat this step.)


------------------------------------------------------------
3. Forced Alignment with MFA
------------------------------------------------------------


Run Montreal Forced Aligner:

    mfa align \
        <corpus_dir> \
        english_us_arpa \
        english_us_arpa \
        <output_dir> \
        --clean

Result:

    mfa_output/
        utt_0.TextGrid
        utt_1.TextGrid
        ...

Each TextGrid contains word-level timestamps:

    word, start_time, end_time


------------------------------------------------------------
4. Batch Processing with mult.py
------------------------------------------------------------


`mult.py` is the main production pipeline.

It:

- Iterates over **all LLM segmentation JSONs**
- Loads the corresponding **TextGrid** file
- Extracts MFA-aligned word timestamps
- Matches every LLM chunk to the aligned words
- For each latency level:
    - low_latency
    - medium_latency
    - high_latency

  it constructs a conservative **1-second streaming timeline**
  and generates bilingual segments.


### 4.1 Alignment Quality Filtering

The script supports skipping bad alignments via:

    allowed_ids = good_ids

(e.g., from CTC confidence analysis or earlier filtering)

Only utterances whose TextGrid name appears in `allowed_ids` are processed.


------------------------------------------------------------
5. Chunk → Word Alignment (LLM → MFA)
------------------------------------------------------------

Each LLM English chunk is normalized and tokenized.

We match chunk tokens to MFA words:

- if a word matches:
    - chunk.start = earliest matched word.start
    - chunk.end   = latest matched word.end
- if no match:
    - chunk.start = None, chunk.end = None

This step yields:

    [
      {"chunk":"which", "start":0.0, "end":0.11},
      {"chunk":"sites", "start":0.11, "end":0.45},
      ...
    ]


------------------------------------------------------------
6. Streaming Emission Timing (Conservative Rule)
------------------------------------------------------------

We generate a timeline of emitted chunks:

Rule:
> Emit a chunk at second S if the chunk has fully finished  
> (chunk.end ≤ S + 1.0)

Example:

    [
      {"second":0, "emit":["which","sites","have","you","been"]},
      {"second":1, "emit":["using"]},
      {"second":2, "emit":["how","did","the"]},
      ...
    ]

This approximates realistic low-latency translation behavior.


------------------------------------------------------------
7. Final Bilingual Segment Construction
------------------------------------------------------------

We map English → Chinese chunks using the LLM JSON:

    eng2zh = {eng_chunk: zh_chunk}

For each second:

- Concatenate emitted English chunks (space-separated)
- Concatenate emitted Chinese chunks (no spaces)

Output:

    {
      "source_low_latency": [...],
      "target_low_latency": [...],
      "source_medium_latency": [...],
      "target_medium_latency": [...],
      "source_high_latency": [...],
      "target_high_latency": [...]
    }


------------------------------------------------------------
8. Output Format (One file per utterance)
------------------------------------------------------------

Each processed utterance produces:

    streaming_dataset/utt_000123.json

with contents like:

    {
      "utt_id": "utt_000123",
      "original_text": "...",
      "source_low_latency": [...],
      "target_low_latency": [...],
      "source_medium_latency": [...],
      "target_medium_latency": [...],
      "source_high_latency": [...],
      "target_high_latency": [...]
    }


------------------------------------------------------------
9. Running the Batch Pipeline
------------------------------------------------------------

Example call:

    python mult.py

Configured with:

    llm_dir       = LLM segmentation directory
    mfa_dir       = MFA TextGrid directory
    output_dir    = dataset output directory
    allowed_ids   = list of reliable TextGrid IDs
    limit         = optional cap on number of files


------------------------------------------------------------
10. Summary
------------------------------------------------------------

This pipeline transforms:

    Raw audio + transcripts
            ↓
       MFA word-level alignment
            ↓
  LLM semantic English/Chinese segmentation
            ↓
  Precise chunk → audio alignment
            ↓
   Conservative second-based streaming segments
            ↓
  Final bilingual streaming translation dataset

The result can be used for:

- Low-latency MT training
- Simultaneous translation modeling
- Speech→text translation
- Alignment-based supervision

End of README
