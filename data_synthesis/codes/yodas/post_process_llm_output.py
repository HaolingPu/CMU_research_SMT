#!/usr/bin/env python3
"""
Post-process LLM segmentation output:
Merge single-word chunks into the previous chunk.

Output to a new directory instead of overwriting.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm


def is_single_word(text):
    """
    Check if a chunk contains only one word (or just punctuation).
    """
    # Remove common punctuation
    cleaned = text.strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    cleaned = cleaned.replace(":", "").replace(";", "").replace("-", "")
    
    # Split by whitespace
    words = cleaned.split()
    
    # Single word or empty (just punctuation)
    return len(words) <= 1


def merge_single_words(eng_list, zh_list):
    """
    Merge single-word English chunks into previous chunk.
    Chinese list is merged accordingly to keep same length.
    
    Args:
        eng_list: List of English chunks
        zh_list: List of Chinese chunks
    
    Returns:
        Tuple of (merged_eng, merged_zh)
    """
    if len(eng_list) != len(zh_list):
        # Lengths don't match - skip merging for safety
        return eng_list, zh_list
    
    if len(eng_list) == 0:
        return eng_list, zh_list
    
    merged_eng = []
    merged_zh = []
    
    i = 0
    while i < len(eng_list):
        current_eng = eng_list[i]
        current_zh = zh_list[i]
        
        # Check if current chunk is single word
        if is_single_word(current_eng) and len(merged_eng) > 0:
            # Merge with previous chunk
            merged_eng[-1] = merged_eng[-1] + " " + current_eng
            merged_zh[-1] = merged_zh[-1] + current_zh  # Chinese no space
        else:
            # Keep as separate chunk
            merged_eng.append(current_eng)
            merged_zh.append(current_zh)
        
        i += 1
    
    return merged_eng, merged_zh


def process_json_file(input_path, output_path):
    """
    Process a single JSON file: merge single-word chunks.
    Save to output_path.
    
    Returns:
        (success, status, modified)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check for error in the JSON
        if 'error' in data:
            # Copy as-is
            shutil.copy2(input_path, output_path)
            return True, "Has error field (copied)", False
        
        modified = False
        
        # Process each latency level
        if "offline" in data:
            levels = ["offline"]
        else:
            levels = ["low_latency", "medium_latency", "high_latency"]

        for level in levels:
            if level not in data:
                continue
            
            eng = data[level].get("English", [])
            zh = data[level].get("Chinese", [])
            
            if not isinstance(eng, list) or not isinstance(zh, list):
                continue
            
            # Merge single words
            merged_eng, merged_zh = merge_single_words(eng, zh)
            
            # Check if anything changed
            if merged_eng != eng or merged_zh != zh:
                modified = True
                data[level]["English"] = merged_eng
                data[level]["Chinese"] = merged_zh
        
        # Save to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, "Modified" if modified else "No change", modified
    
    except Exception as e:
        return False, f"Error: {str(e)}", False


def process_language(input_dir, output_dir, lang):
    """
    Process all JSON files for a language.
    Save to output_dir while preserving directory structure.
    """
    print(f"\n{'='*60}")
    print(f"Post-processing LLM output: {lang}")
    print(f"Input:  {input_dir}/{lang}")
    print(f"Output: {output_dir}/{lang}")
    print(f"{'='*60}\n")
    
    input_lang_dir = os.path.join(input_dir, lang)
    output_lang_dir = os.path.join(output_dir, lang)
    
    if not os.path.exists(input_lang_dir):
        print(f"❌ Input directory not found: {input_lang_dir}")
        return
    
    # Create output language directory
    os.makedirs(output_lang_dir, exist_ok=True)
    
    # Get all parquet directories (8-digit folders)
    parquet_dirs = sorted([
        d for d in os.listdir(input_lang_dir)
        if os.path.isdir(os.path.join(input_lang_dir, d)) and d.isdigit() and len(d) == 8
    ])
    
    print(f"Found {len(parquet_dirs)} parquet directories\n")
    
    total_files = 0
    modified_files = 0
    error_files = 0
    
    for parquet_name in parquet_dirs:
        input_parquet_dir = os.path.join(input_lang_dir, parquet_name)
        output_parquet_dir = os.path.join(output_lang_dir, parquet_name)
        
        # Create output parquet directory
        os.makedirs(output_parquet_dir, exist_ok=True)
        
        json_files = sorted([f for f in os.listdir(input_parquet_dir) if f.endswith('.json')])
        
        print(f"📁 Processing {parquet_name} ({len(json_files)} files)")
        
        parquet_modified = 0
        
        for fname in tqdm(json_files, desc=f"  {parquet_name}"):
            input_path = os.path.join(input_parquet_dir, fname)
            output_path = os.path.join(output_parquet_dir, fname)
            
            success, status, is_modified = process_json_file(input_path, output_path)
            total_files += 1
            
            if success:
                if is_modified:
                    print(f"modified: ,{input_path}")
                    modified_files += 1
                    parquet_modified += 1
            else:
                error_files += 1
        
        print(f"  ✓ Modified {parquet_modified}/{len(json_files)} files\n")
    
    print(f"{'='*60}")
    print(f"✅ Post-processing complete!")
    print(f"{'='*60}")
    print(f"Total files processed: {total_files}")
    print(f"Files modified:        {modified_files}")
    print(f"Files with errors:     {error_files}")
    print(f"Files unchanged:       {total_files - modified_files - error_files}")
    print(f"\n📁 Output saved to: {output_dir}/{lang}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process LLM segmentation: merge single-word chunks"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory of LLM output (e.g.,  /data/user_data/haolingp/outputs/llm_output)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for modified files (e.g., /data/user_data/haolingp/outputs/llm_output_modified)"
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to process (e.g., en000)"
    )
    
    args = parser.parse_args()
    
    process_language(args.input_dir, args.output_dir, args.lang)


if __name__ == "__main__":
    main()