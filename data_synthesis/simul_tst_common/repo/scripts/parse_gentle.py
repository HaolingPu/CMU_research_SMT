import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm


def remove_characters(sentence):
    """
    Remove punctuation that is not used for word-level matching.
    """
    remove_chars = "!$?%"
    trans_table = str.maketrans("", "", remove_chars)
    cleaned_text = sentence.translate(trans_table)
    words = cleaned_text.split()
    return words


def word_list(words):
    """
    Normalize words for matching with Gentle output.

    This function splits words that contain punctuation such as:
    - comma
    - period
    - hyphen

    Example:
        "1,000" -> ["1", "000"]
        "well-known" -> ["well", "known"]
    """
    tmp = []
    symbols = ".,-"

    for word in words:
        if not word:
            continue

        if word[-1] in symbols:
            word = word.rstrip(symbols)

            if any(symbol in word for symbol in symbols):
                split_words = re.split(r"[,\-.]", word)
                split_words = [w for w in split_words if w]
                tmp.extend(split_words)
            else:
                tmp.append(word)

        elif any(symbol in word for symbol in symbols):
            split_words = re.split(r"[,\-.]", word)
            split_words = [w for w in split_words if w]
            tmp.extend(split_words)

        else:
            tmp.append(word)

    return tmp


def normalize_transcript(transcript):
    """
    Convert one transcript line into a normalized word list.
    """
    words = remove_characters(transcript)
    return word_list(words)


def get_offset_and_duration(w_sentence, tmp_words):
    """
    Get offset and duration for one sentence.

    If the first word is aligned, offset can be obtained.
    If the last word is aligned, duration can be computed.
    Otherwise, missing values are returned as None.
    """
    first_word = w_sentence[0]
    last_word = w_sentence[-1]

    first_matches = first_word.get("word") == tmp_words[0]
    last_matches = last_word.get("word") == tmp_words[-1]

    first_success = first_word.get("case") == "success"
    last_success = last_word.get("case") == "success"

    if first_success and first_matches:
        offset = first_word["start"]

        if last_success and last_matches:
            duration = last_word["end"] - first_word["start"]
            auto = True
        else:
            duration = None
            auto = False

    else:
        offset = None
        duration = None
        auto = False

    return offset, duration, auto


def parse_gentle_json(json_file, txt_dir, output_dir):
    """
    Parse one Gentle JSON file and create one YAML file.
    """
    file_name = json_file.stem
    txt_file = txt_dir / f"{file_name}.txt"
    output_file = output_dir / f"{file_name}.yaml"

    if not txt_file.exists():
        print(f"Text file does not exist. Skipping: {txt_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        gentle_data = json.load(f)

    data = gentle_data["words"]
    transcripts = gentle_data["transcript"].replace("\r", "").split("\n")
    transcripts = [line for line in transcripts if line.strip()]

    all_words = []

    for transcript in transcripts:
        tmp_words = normalize_transcript(transcript)
        all_words.extend(tmp_words)

    assert len(data) == len(all_words), (
        f"Word count mismatch in {json_file.name}: "
        f"Gentle words = {len(data)}, transcript words = {len(all_words)}"
    )

    offsets = []
    durations = []
    autos = []

    begin = 0

    for transcript in transcripts:
        tmp_words = normalize_transcript(transcript)
        length = len(tmp_words)

        if length == 0:
            continue

        w_sentence = data[begin:begin + length]

        offset, duration, auto = get_offset_and_duration(
            w_sentence=w_sentence,
            tmp_words=tmp_words
        )

        offsets.append(offset)
        durations.append(duration)
        autos.append(auto)

        begin += length

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]

    assert len(offsets) == len(durations)
    assert len(offsets) == len(lines), (
        f"Sentence count mismatch in {file_name}: "
        f"offsets = {len(offsets)}, txt lines = {len(lines)}"
    )

    talk_id = file_name.split("_")[-1]

    yaml_lines = []

    for offset, duration, auto in zip(offsets, durations, autos):
        if offset is not None and duration is not None:
            line = (
                f"- {{duration: {duration:.6f}, offset: {offset:.6f}, "
                f"speaker_id: spk.{talk_id}, wav: ted_{talk_id}.wav, auto: {auto}}}"
            )

        elif offset is not None and duration is None:
            line = (
                f"- {{duration: null, offset: {offset:.6f}, "
                f"speaker_id: spk.{talk_id}, wav: ted_{talk_id}.wav, auto: False}}"
            )

        else:
            line = (
                f"- {{duration: null, offset: null, "
                f"speaker_id: spk.{talk_id}, wav: ted_{talk_id}.wav, auto: False}}"
            )

        yaml_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in yaml_lines:
            f.write(line + "\n")

    print(f"Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gentle JSON alignment files into YAML files."
    )

    parser.add_argument(
        "--json_dir",
        required=True,
        help="Directory containing Gentle JSON files."
    )

    parser.add_argument(
        "--txt_dir",
        required=True,
        help="Directory containing transcript TXT files."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where YAML files will be saved."
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return

    for json_file in tqdm(json_files):
        parse_gentle_json(
            json_file=json_file,
            txt_dir=txt_dir,
            output_dir=output_dir
        )

    print("finish")


if __name__ == "__main__":
    main()