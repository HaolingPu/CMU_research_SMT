import json

def filter_metricx(
    input_file,
    output_file,
    threshold=5.0,
):
    """
    Filter out samples where prediction > threshold.
    """
    kept = 0
    removed = 0

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            ex = json.loads(line)

            # Skip if prediction > threshold
            if ex["prediction"] > threshold:
                removed += 1
                continue

            fout.write(json.dumps(ex) + "\n")
            kept += 1

    print("=== MetricX Filtering Done ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Threshold: {threshold}")
    print(f"Kept samples: {kept}")
    print(f"Removed samples: {removed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MetricX prediction jsonl")
    parser.add_argument("--output", required=True, help="Filtered jsonl")
    parser.add_argument("--threshold", type=float, default=5.0)

    args = parser.parse_args()

    filter_metricx(args.input, args.output, args.threshold)
