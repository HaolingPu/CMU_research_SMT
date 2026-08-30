from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

# The local lightweight test interpreter does not install runtime ML packages.
pandas = types.ModuleType("pandas")
pandas.DataFrame = object
sys.modules.setdefault("pandas", pandas)
transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers)

import consensus_decoding_token_id_level_instruct as decoder  # noqa: E402


class FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return [10, 20, 30] if kwargs["tokenize"] else "rendered-no-think"


class ConsensusPromptHelpersTest(unittest.TestCase):
    def test_probe_disables_thinking(self) -> None:
        tokenizer = FakeTokenizer()
        ids = decoder.build_translation_probe_prompt_prefix_token_ids(
            tokenizer,
            "The bank",
            has_target_prefix=False,
        )
        self.assertEqual(ids, [10, 20, 30])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_final_completion_disables_thinking(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = decoder.build_final_completion_prompt(
            tokenizer,
            "The bank collapsed",
            committed_text="prefix",
        )
        self.assertEqual(prompt, "rendered-no-thinkprefix")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_meta_and_long_futures_are_rejected(self) -> None:
        self.assertFalse(decoder.is_valid_future_text("The prompt says to continue here"))
        self.assertFalse(decoder.is_valid_future_text("word " * 21))
        self.assertTrue(decoder.is_valid_future_text("collapsed after the heavy rain"))

    def test_diversity_filter_caps_repeated_openings_and_near_duplicates(self) -> None:
        selected = decoder.select_diverse_futures([
            "crucial for training the model",
            "crucial for training a larger model",
            "crucial for training the final model",
            "important for understanding the historical setting",
            "important for understanding this historical setting",
            "inevitably repetitive and less useful than intended",
        ])
        self.assertEqual(sum(text.startswith("crucial") for text in selected), 1)
        self.assertEqual(sum(text.startswith("important") for text in selected), 1)
        self.assertIn("inevitably repetitive and less useful than intended", selected)

    def test_coordinated_set_audits_model_and_filter_status(self) -> None:
        response = {"choices": [{"text": (
            "1. crucial for training models\n"
            "2. crucial for testing systems\n"
            "3. brief but historically grounded\n"
            "4. unclear\n"
        )}]}
        with patch.object(decoder, "_http_json", return_value=response):
            futures, infos, audit = decoder._sample_coordinated_future_set(
                sampler_tokenizer=FakeTokenizer(),
                observed_source="These introductions are",
                committed_text="",
                target_lang="Chinese",
                num_futures=4,
                api_base="http://localhost:1/v1",
                api_model="gemma4-sampler",
                api_timeout=1.0,
                sample_temperature=1.0,
                top_p=0.98,
                max_tokens=40,
            )
        self.assertEqual(len(futures), 2)
        self.assertTrue(all(info["model"] == "gemma4-sampler" for info in infos))
        self.assertEqual([info["mode"] for info in infos], ["plausible", "contrastive"])
        self.assertEqual([item["accepted"] for item in audit], [True, False, True, False])
        self.assertEqual(audit[-1]["reason"], "too_short")

        lines = decoder.format_raw_future_groups(audit)
        self.assertIn("[Raw candidates] Gemma 4 |", lines[0])
        self.assertEqual(sum("model=gemma4-sampler" in line for line in lines), 2)
        self.assertIn("mode=plausible", lines[0])
        self.assertFalse(any("status=" in line for line in lines))
        self.assertTrue(any("Filter summary: kept=1/2" in line for line in lines))

        selected = decoder.format_selected_future_groups(futures, infos)
        self.assertIn("[Selected candidates] Gemma 4 |", selected[0])
        self.assertEqual(sum("mode=" in line for line in selected), 2)


if __name__ == "__main__":
    unittest.main()
