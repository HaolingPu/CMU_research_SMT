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
pandas.isna = lambda value: value is None or (isinstance(value, float) and value != value)
sys.modules.setdefault("pandas", pandas)
transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers)

import consensus_decoding_token_id_level_instruct as decoder  # noqa: E402


class PieceTokenizer:
    """Token id i decodes to PIECES[i]."""

    PIECES = ["Kub", "ernetes", "边缘", "CDC", "）", "<|im_end|>", "*", "<think>", "</tool_call>", "<|fim_prefix|>", " B1"]
    all_special_ids = [5]

    def decode(self, ids, **kwargs):
        return "".join(self.PIECES[i] for i in ids)


class LatinTokenRuleTest(unittest.TestCase):
    def tearDown(self) -> None:
        decoder.BLOCK_LATIN_TOKENS = False

    def test_default_allows_english_terms_in_vote_and_commit(self):
        tokenizer = PieceTokenizer()
        kept, debug = decoder.filter_distribution_token_ids(tokenizer, {0: 0.6, 2: 0.3, 3: 0.1}, {"reason": "ok"})
        self.assertEqual(set(kept), {0, 2, 3})
        self.assertEqual(debug["filtered_disallowed_tokens"], [])
        kept_ids, removed = decoder.sanitize_pending_token_ids(tokenizer, [0, 1, 4])
        self.assertEqual(kept_ids, [0, 1, 4])
        self.assertEqual(removed, [])

    def test_block_flag_restores_the_old_rule(self):
        decoder.BLOCK_LATIN_TOKENS = True
        tokenizer = PieceTokenizer()
        kept, _ = decoder.filter_distribution_token_ids(tokenizer, {0: 0.6, 2: 0.3, 3: 0.1}, {"reason": "ok"})
        self.assertEqual(set(kept), {2})
        kept_ids, removed = decoder.sanitize_pending_token_ids(tokenizer, [0, 2, 3])
        self.assertEqual(kept_ids, [2])
        self.assertEqual([item["reason"] for item in removed], ["ascii_letters", "ascii_letters"])

    def test_other_filters_still_apply(self):
        tokenizer = PieceTokenizer()
        self.assertEqual(decoder._disallowed_generation_token_reason(tokenizer, 5), "special_token_id")
        self.assertEqual(decoder._disallowed_generation_token_reason(tokenizer, 6), "markdown_token")

    def test_added_tag_tokens_stay_blocked(self):
        tokenizer = PieceTokenizer()
        for tok_id in (7, 8, 9):
            self.assertEqual(decoder._disallowed_generation_token_reason(tokenizer, tok_id), "special_token_text")
        self.assertIsNone(decoder._disallowed_generation_token_reason(tokenizer, 10))

    def test_completion_keeps_the_space_between_english_words(self):
        class ChatTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

        def completion_of(committed, text):
            reply = {"choices": [{"text": text, "finish_reason": "stop"}]}
            with patch.object(decoder, "_http_json", return_value=reply):
                return decoder.force_complete_translation(ChatTokenizer(), "We use Visual Studio Code.", committed,
                                                          "http://unused", "model")
        self.assertEqual(completion_of("我们用 Visual", " Studio Code。"), " Studio Code。")
        self.assertEqual(completion_of("我们用", " Visual Studio Code。"), "Visual Studio Code。")
        self.assertEqual(completion_of("我们用 Visual", "工作室。"), "工作室。")
        self.assertEqual(completion_of("", " 我们。"), "我们。")

    def test_command_line_default_and_flag(self):
        base = ["decoder", "--instruct-tokenizer-path", "unused", "--instruct-api-base", "http://unused"]
        with patch.object(sys, "argv", base):
            self.assertFalse(decoder.parse_args().block_latin_tokens)
        with patch.object(sys, "argv", base + ["--block-latin-tokens"]):
            self.assertTrue(decoder.parse_args().block_latin_tokens)


if __name__ == "__main__":
    unittest.main()
