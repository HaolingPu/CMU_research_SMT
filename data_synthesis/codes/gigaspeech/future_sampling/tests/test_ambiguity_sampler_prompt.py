from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from ambiguity_sampler_prompt import (  # noqa: E402
    PROMPT_VERSION,
    build_coordinated_future_messages,
)


class CoordinatedFuturePromptTest(unittest.TestCase):
    def test_prompt_requests_one_natural_coordinated_set(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="The bank",
            target_lang="Chinese",
            committed_text="",
            num_candidates=10,
        )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertEqual(PROMPT_VERSION, "future_set_v1")
        self.assertIn("Observed English prefix:\nThe bank", prompt)
        self.assertIn("exactly 10 continuations", prompt)
        self.assertIn("Plan the complete set", prompt)
        self.assertIn("mutually distinct", prompt)
        self.assertNotIn("plausible mode", prompt.lower())
        self.assertNotIn("contrastive", prompt.lower())
        self.assertNotIn("These introductions are", prompt)

    def test_committed_translation_is_included(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="The bank",
            target_lang="Chinese",
            committed_text="this-prefix",
            num_candidates=10,
        )
        self.assertIn("this-prefix", messages[1]["content"])

    def test_invalid_requests_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_coordinated_future_messages(
                observed_source="",
                target_lang="Chinese",
                committed_text="",
                num_candidates=10,
            )


if __name__ == "__main__":
    unittest.main()
