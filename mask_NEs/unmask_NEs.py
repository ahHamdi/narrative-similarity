import json
from pathlib import Path

FILE_MASKED = "data/dev_track_a_NE_masked.jsonl"
FILE_ORIGINAL = "data/dev_track_a.jsonl"
OUTPUT_FILE = "output/dev_track_a_restored.jsonl"

TEXT_FIELDS = {"anchor_text", "text_a", "text_b", "text"}


def restore_jsonl(masked_path, original_path, output_path):
    masked_path = Path(masked_path)
    original_path = Path(original_path)
    output_path = Path(output_path)

    with masked_path.open("r", encoding="utf-8") as f_masked, \
         original_path.open("r", encoding="utf-8") as f_orig, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_id, (line_masked, line_orig) in enumerate(
            zip(f_masked, f_orig), start=1
        ):
            if not line_masked.strip() or not line_orig.strip():
                continue

            data_masked = json.loads(line_masked)
            data_orig = json.loads(line_orig)

            restored = {}

            # Restore NEs from the original text
            for field in TEXT_FIELDS:
                if field in data_masked:
                    if field not in data_orig:
                        raise ValueError(
                            f"Line {line_id}: field '{field}' missing in original file"
                        )
                    restored[field] = data_orig[field]

            for field, value in data_masked.items():
                if field not in TEXT_FIELDS:
                    restored[field] = value

            fout.write(json.dumps(restored, ensure_ascii=False) + "\n")

    print(f"Unmasking NEs finished : {output_path}")


if __name__ == "__main__":
    restore_jsonl(FILE_MASKED, FILE_ORIGINAL, OUTPUT_FILE)
