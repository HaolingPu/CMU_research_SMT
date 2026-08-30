import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import analyze_ambiguity_pilot as analyzer  # noqa: E402


class AnalyzeAmbiguityPilotTest(unittest.TestCase):
    def test_case_report_parses_futures_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "case.json"
            verbose = root / "verbose_case.log"
            result.write_text(json.dumps({
                "utt_id": "case",
                "source_full_text": "The bank",
                "prediction": "银行",
                "reference_text": "河岸",
                "actions": ["READ", "WRITE"],
                "target_trajectory": ["", "银行"],
                "metrics": {"bleu_char": 1.0, "laal_text": 2.0},
            }), encoding="utf-8")
            verbose.write_text(
                "Chunk 1/2\n"
                "future_source_prefix: 'The bank'\n"
                "future[0] model=gemma4-sampler mode=plausible: 'approved the loan'\n"
                "future[1] model=qwen38-sampler mode=contrastive: 'collapsed after rain'\n"
                "[Step 6-7] commit_after_trim=''\n"
                "-> READ (too few futures)\n",
                encoding="utf-8",
            )
            case = analyzer.analyze_case(result, verbose)
            self.assertEqual(case["future_count"], 2)
            self.assertEqual(case["too_few_chunks"], 1)
            self.assertEqual(case["meta_leakage"], [])
            self.assertEqual(case["chunks"][0]["action"], "READ")
            self.assertEqual(case["chunks"][0]["futures"][0]["model"], "gemma4-sampler")


if __name__ == "__main__":
    unittest.main()
