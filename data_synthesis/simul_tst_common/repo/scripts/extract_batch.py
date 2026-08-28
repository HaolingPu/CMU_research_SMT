import argparse
import json
import os


def parse_model_content(content):
    """
    Parse the JSON string returned by the model.

    Expected format:
    {
        "segmented_pairs": [[English, Translation], ...],
        "output": Translation
    }
    """
    content = content.replace("\n", "")

    return json.loads(content)


def extract_outputs(input_file_path):
    rewrite_text = []
    sep_en = []
    sep_tgt = []

    with open(input_file_path, "r", encoding="utf-8") as infile:
        for line_id, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                content = data["response"]["body"]["choices"][0]["message"]["content"]
                parsed_content = parse_model_content(content)

                output = parsed_content["output"]
                segment_pairs = parsed_content["segmented_pairs"]

                rewrite_text.append(output)

                src_segments = []
                tgt_segments = []

                for pair in segment_pairs:
                    src_segments.append(pair[0])
                    tgt_segments.append(pair[1])

                sep_en.append(" / ".join(src_segments))
                sep_tgt.append(" / ".join(tgt_segments))

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line {line_id}: {e}")

            except KeyError as e:
                print(f"Skipping line {line_id}: missing key {e}")

            except Exception as e:
                print(f"Skipping line {line_id}: {e}")

    return rewrite_text, sep_en, sep_tgt


def write_lines(output_path, lines):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract outputs and segmented pairs from OpenAI batch output JSONL."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the batch output JSONL file."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output files will be saved."
    )

    parser.add_argument(
        "--target_lang",
        default="tgt",
        help="Target language suffix for segmented output file, e.g., ja, de, zh. Default: tgt."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rewrite_text, sep_en, sep_tgt = extract_outputs(args.input)

    write_lines(
        os.path.join(args.output_dir, "Simul-LLM.{target_language}"),
        rewrite_text
    )

    write_lines(
        os.path.join(args.output_dir, "sep.en"),
        sep_en
    )

    write_lines(
        os.path.join(args.output_dir, f"sep.{args.target_lang}"),
        sep_tgt
    )

    print("finish")


if __name__ == "__main__":
    main()
