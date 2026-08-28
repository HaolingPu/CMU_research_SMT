import argparse
import re


def split_sentences(text):
    """
    Split text into sentences when . or ? is followed by whitespace.
    Also handles .” and ?”.
    """
    text = re.sub(r'([.?]”?)\s+', r'\1\n', text)
    return [s.strip() for s in text.splitlines() if s.strip()]


def is_sentence_end(line):
    """
    Check whether the line ends with sentence-final punctuation.
    """
    return line.endswith((".", "?", ".”", "?”"))


def process_file(input_path, output_path):
    sentences = []
    buffer = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            buffer.append(line)

            if is_sentence_end(line):
                merged_text = " ".join(buffer)
                split_lines = split_sentences(merged_text)
                sentences.extend(split_lines)
                buffer = []

    # If there is remaining text without final punctuation, keep it.
    if buffer:
        merged_text = " ".join(buffer)
        split_lines = split_sentences(merged_text)
        sentences.extend(split_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print("finish")


def main():
    parser = argparse.ArgumentParser(
        description="Merge broken ASR lines and split text into one sentence per line."
    )
    parser.add_argument("--input", required=True, help="Path to input text file.")
    parser.add_argument("--output", required=True, help="Path to output text file.")
    args = parser.parse_args()

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()