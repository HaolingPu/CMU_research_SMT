import argparse
import hashlib
import json
import os
from tqdm import tqdm


def create_hash(request_id):
    """Create a unique hash ID for each input sentence."""
    return hashlib.md5(request_id.encode("utf-8")).hexdigest()


def get_language_name(lang_pair):
    """Convert language pair to target language name."""
    lang = lang_pair.split("-")[1]

    if lang == "ja":
        return "Japanese"
    elif lang == "zh":
        return "Chinese"
    elif lang == "de":
        return "German"
    else:
        raise ValueError(f"Unsupported language: {lang}")


def get_lang_pair_and_data_type(file_path):
    """
    Extract language pair and data type from a MuST-C-style file path.

    Example:
    /project/nlp-cache/data/MuST-C/en-ja/data/train/txt/train.en

    lang_pair: en-ja
    data_type: train
    """
    parts = file_path.split("/")

    lang_pair = parts[5]
    data_type = parts[7]

    return lang_pair, data_type


def create_request_data(text, language):
    """Create one JSONL request item."""
    return {
        "custom_id": create_hash(text),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-05-13",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You will be provided with a sentence in English, "
                        f"and your task is to interpret it into {language}. "
                        f"Always answer in the following JSON format: "
                        f"{{'segmented_pairs': List[Tuple[English, {language}]], "
                        f"'output': {language}}}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Instructions:\n"
                        "'Salami Technique' in simultaneous interpretation refers to "
                        "a technique where the interpreter breaks down the source language "
                        "input into smaller, manageable segments that each contain enough "
                        "information to be accurately interpreted.\n"
                        "1. Break down the following sentence into smaller segments for "
                        "easier simultaneous interpretation.\n"
                        f"2. Translate each segment into {language}.\n"
                        "3. Connect the translated segments.\n"
                        "----------------------\n"
                        f"Inputs:\n{text}\n"
                    ),
                },
            ],
            "temperature": 0.5,
            "top_p": 0.0,
            "seed": 0,
            "n": 1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        },
    }


def process_file(file_path, output_base_dir, max_lines_per_file):
    """Convert one input text file into one or more JSONL files."""
    lang_pair, data_type = get_lang_pair_and_data_type(file_path)
    language = get_language_name(lang_pair)

    output_dir = os.path.join(output_base_dir, lang_pair, data_type)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    jsonl_base_name = os.path.splitext(base_name)[0] + "_part"

    with open(file_path, "r", encoding="utf-8") as f:
        unique_lines = list({line.strip() for line in f if line.strip()})

    file_index = 1
    line_counter = 0
    output_file = None

    for text in unique_lines:
        if line_counter % max_lines_per_file == 0:
            if output_file is not None:
                output_file.close()

            jsonl_file_name = f"{jsonl_base_name}{file_index}.jsonl"
            jsonl_file_path = os.path.join(output_dir, jsonl_file_name)
            output_file = open(jsonl_file_path, "w", encoding="utf-8")

            file_index += 1

        request_data = create_request_data(text, language)
        output_file.write(json.dumps(request_data, ensure_ascii=False) + "\n")

        line_counter += 1

    if output_file is not None:
        output_file.close()

    print(f"Finished: {file_path}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL files for OpenAI batch processing."
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input text file path(s). You can specify multiple files.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory.",
    )

    parser.add_argument(
        "--max_lines_per_file",
        type=int,
        default=50000,
        help="Maximum number of lines per JSONL file. Default: 50000.",
    )

    args = parser.parse_args()

    for file_path in tqdm(args.input):
        process_file(
            file_path=file_path,
            output_base_dir=args.output_dir,
            max_lines_per_file=args.max_lines_per_file,
        )

    print("finish")


if __name__ == "__main__":
    main()
