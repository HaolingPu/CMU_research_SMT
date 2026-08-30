from pathlib import Path
import sys
import types
import unittest


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


if __name__ == "__main__":
    unittest.main()
