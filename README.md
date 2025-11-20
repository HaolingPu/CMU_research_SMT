# Granary Forced Alignment + LLM Segmentation Pipeline

This pipeline converts raw audio+text stored in Parquet files into 
second-level streaming translation segments using:

- Montreal Forced Aligner (MFA)
- LLM-generated English/Chinese segmentation
- Audio–text alignment via TextGrid
- Second-based conservative emission rules

The full process is outlined below.

------------------------------------------------------------
1. Load audio and text from Parquet
------------------------------------------------------------

The dataset is stored in HF-style Parquet format:

- Each entry contains:
  - audio.bytes  (raw WAV bytes)
  - text         (transcription)

The pipeline:
- Reads the Parquet file using pandas
- Extracts raw audio bytes from each row
- Decodes audio using soundfile
- Normalizes / cleans text for alignment

Output of this step:  
In-memory audio + text for each utterance.


------------------------------------------------------------
2. Export a corpus compatible with MFA
------------------------------------------------------------

For each utterance, we generate:

- utt_X.wav  (decoded from bytes)
- utt_X.lab  (cleaned text)

These are placed in:

    mfa_corpus/
        utt_0.wav
        utt_0.lab
        utt_1.wav
        utt_1.lab
        ...

This is the standard corpus format required by Montreal Forced Aligner.


------------------------------------------------------------
3. Run MFA to obtain forced alignment
------------------------------------------------------------

MFA is used to produce word-level (and phone-level) timestamps.

Command:

    mfa align \
        <corpus_dir> \
        english_us_arpa \
        english_us_arpa \
        <output_dir> \
        --clean

This creates TextGrid alignment files:

    mfa_output/
        utt_0.TextGrid
        utt_1.TextGrid
        ...

Each TextGrid contains:
- "words" tier with (word, start_time, end_time)
- "phones" tier


------------------------------------------------------------
4. Extract timestamps from TextGrid
------------------------------------------------------------

We use the TGT library to parse TextGrid:

- Load the "words" tier
- Remove empty words
- Produce a list of dictionaries:

    [
      {"word": "which", "start": 0.00, "end": 0.11},
      {"word": "sites", "start": 0.11, "end": 0.45},
      ...
    ]

These word-level timestamps are the ground-truth alignment.


------------------------------------------------------------
5. Align LLM segmentation chunks to MFA words
------------------------------------------------------------

The LLM JSON segmentation provides English & Chinese chunks:

    {
      "low_latency": {
        "English": [...],
        "Chinese": [...]
      }
    }

For each English chunk:
- Normalize text (lowercase, strip punctuation)
- Tokenize chunk
- Match tokens to MFA word list
- Compute chunk start/end time:

      start = earliest matched word start
      end   = latest matched word end

Output example:

    {"chunk": "which", "start": 0.0, "end": 0.11}
    {"chunk": "sites", "start": 0.11, "end": 0.45}
    ...


------------------------------------------------------------
6. Build second-level streaming emission timeline
------------------------------------------------------------

We apply a *conservative* rule:

    A chunk is emitted at second S iff its end_time <= S + 1.0

This produces:

    [
      {"second": 0, "emit": ["which", "sites", "have", ...]},
      {"second": 1, "emit": ["using", ...]},
      {"second": 2, "emit": ["how", "did", ...]},
      ...
    ]

This mirrors a realistic low-latency streaming translation scenario.


------------------------------------------------------------
7. Construct final bilingual streaming segments
------------------------------------------------------------

We map English chunks → Chinese chunks using the LLM segmentation:

    eng_to_zh = { eng_chunk : zh_chunk }

For each second:
- Concatenate English chunks into one string
- Concatenate corresponding Chinese chunks
- Produce aligned source–target segments:

    {
      "source": ["which sites have you been ...", ...],
      "target": ["你最近使用了哪些网站……", ...]
    }

This can be used as training data for streaming MT, 
speech-to-text translation, or simultaneous translation models.


------------------------------------------------------------
Pipeline Summary
------------------------------------------------------------

1. Load Parquet audio + text  
2. Convert raw bytes → WAV  
3. Generate MFA-style corpus  
4. Run MFA forced alignment  
5. Extract word timestamps  
6. Align LLM chunks to MFA  
7. Emit per-second streaming segments  
8. Produce final bilingual training pairs

This pipeline creates high-quality, time-aligned, 
low-latency translation trajectories from raw audio.

------------------------------------------------------------

End of README
