from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from ambiguity_sampler_prompt import (  # noqa: E402
    PROMPT_VERSION,
    build_ambiguity_sampler_messages,
)


class AmbiguitySamplerPromptTest(unittest.TestCase):
    def test_prompt_is_general_and_contains_icl_examples(self) -> None:
        messages = build_ambiguity_sampler_messages(
            target_lang="Chinese",
            committed_text="",
            sampling_mode="contrastive",
        )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertEqual(PROMPT_VERSION, "ambiguity_icl_v2")
        self.assertIn('Partial: "The bank"', prompt)
        self.assertIn('Partial: "The agreement', prompt)
        self.assertIn("translation", prompt)
        self.assertNotIn("five-axis", prompt.lower())
        self.assertNotIn("axis (a)", prompt.lower())
        self.assertIn("unsupported change to a technical topic", prompt)
        self.assertIn("vary the first content word", prompt)

    def test_committed_translation_is_included(self) -> None:
        messages = build_ambiguity_sampler_messages(
            target_lang="Chinese",
            committed_text="this-prefix",
            sampling_mode="plausible",
        )
        self.assertIn("this-prefix", messages[1]["content"])

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_ambiguity_sampler_messages(
                target_lang="Chinese",
                committed_text="",
                sampling_mode="axis5",
            )


if __name__ == "__main__":
    unittest.main()
