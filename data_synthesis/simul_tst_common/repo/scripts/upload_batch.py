
import argparse
import os
from openai import OpenAI

OPENAI_ORGANIZATION_KEY='your_organization_key'
OPENAI_API_KEY='your_api_key'
client = OpenAI(organization=OPENAI_ORGANIZATION_KEY, api_key=OPENAI_API_KEY)

def create_batch(client, file_path, description):
    """Upload one JSONL file and create a batch job."""
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )

    return batch


def process_files(
    client,
    input_dir,
    lang_pairs,
    file_names,
    file_numbers,
    description,
):
    for lang_pair in lang_pairs:
        for file_name in file_names:
            for num in file_numbers:
                file_path = os.path.join(
                    input_dir,
                    lang_pair,
                    file_name,
                    f"{lang_pair}_{file_name}_part{num}.jsonl"
                )

                if not os.path.exists(file_path):
                    print(f"File does not exist. Skipping: {file_path}")
                    continue

                batch = create_batch(
                    client=client,
                    file_path=file_path,
                    description=description
                )

                print(
                    f"Batch created successfully: "
                    f"{lang_pair}, {file_name}, part{num}, batch_id={batch.id}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Upload JSONL files and create OpenAI batch jobs."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Base input directory containing JSONL files."
    )

    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        required=True,
        help="Language pairs, e.g., en-ja en-de en-zh."
    )

    parser.add_argument(
        "--file_names",
        nargs="+",
        required=True,
        help="Dataset names, e.g., train dev tst-COMMON."
    )

    parser.add_argument(
        "--file_numbers",
        nargs="+",
        type=int,
        required=True,
        help="Part numbers, e.g., 1 2 3 4 5 6 7."
    )

    parser.add_argument(
        "--description",
        default="batch job",
        help="Description stored in the batch metadata."
    )

    parser.add_argument(
        "--organization",
        default=None,
        help="OpenAI organization ID. Optional."
    )

    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=args.organization
    )

    process_files(
        client=client,
        input_dir=args.input_dir,
        lang_pairs=args.lang_pairs,
        file_names=args.file_names,
        file_numbers=args.file_numbers,
        description=args.description,
    )

    print("finish")


if __name__ == "__main__":
    main()