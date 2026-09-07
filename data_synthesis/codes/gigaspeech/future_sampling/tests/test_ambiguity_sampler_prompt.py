from pathlib import Path
import re
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from ambiguity_sampler_prompt import (  # noqa: E402
    AMBIGUITY_ICL_EXAMPLES,
    PROMPT_VERSION,
    SUFFIX_ICL_PROMPT_VERSION,
    build_coordinated_future_messages,
    parse_grouped_future_output,
    sample_grouped_futures,
)


class CoordinatedFuturePromptTest(unittest.TestCase):
    def test_prompt_requests_two_jointly_planned_groups(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="The bank",
            target_lang="Chinese",
            committed_text="",
            num_candidates=20,
        )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertEqual(PROMPT_VERSION, "future_set_v2_two_groups")
        self.assertIn("Observed English prefix:\nThe bank", prompt)
        self.assertIn("exactly 20 continuations", prompt)
        self.assertIn("first 10 plausible candidates", prompt)
        self.assertIn("then 10 contrastive candidates", prompt)
        self.assertIn("Plan the complete set", prompt)
        self.assertIn("mutually distinct", prompt)
        self.assertIn("Plausible\n1.", prompt)
        self.assertIn("Contrastive\n11.", prompt)
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
        with self.assertRaises(ValueError):
            build_coordinated_future_messages(
                observed_source="The bank",
                target_lang="Chinese",
                committed_text="",
                num_candidates=9,
            )

    def test_v3_prioritizes_suffix_fit_and_allows_shorter_groups(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="A fresh prefix", target_lang="Chinese", committed_text="",
            num_candidates=20, prompt_version=SUFFIX_ICL_PROMPT_VERSION,
        )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("at most 20 candidates total", prompt)
        self.assertIn("up to 10 Plausible and up to 10 Contrastive", prompt)
        self.assertIn("Never pad a list", prompt)
        self.assertIn("Continue the unfinished sentence", prompt)
        self.assertIn("PREFIX + one space + SUFFIX", prompt)
        self.assertIn("It is okay to share necessary opening words", prompt)
        self.assertIn("For an empty group write None", prompt)
        self.assertNotIn("exactly 20", prompt)
        self.assertNotIn("already committed", prompt)
        self.assertIn("Observed English prefix:\nA fresh prefix", messages[1]["content"])

    def test_v3_uses_generic_positive_and_negative_examples(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="A fresh prefix", target_lang="Chinese", committed_text="",
            num_candidates=4, prompt_version=SUFFIX_ICL_PROMPT_VERSION,
        )
        system, user = (message["content"] for message in messages)
        self.assertIn("VALID suffix: bowl of soup beside the bread.", system)
        self.assertIn("INVALID suffix: she waved to the crowd.", system)
        self.assertIn('leaves "the tools were" unfinished', system)
        self.assertIn("up to 2 per group", user)
        self.assertNotIn("fork", system)
        self.assertNotIn("giant", system)
        self.assertNotIn("1015", system)

    def test_unknown_prompt_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_coordinated_future_messages(
                observed_source="The bank", target_lang="Chinese", committed_text="",
                num_candidates=20, prompt_version="typo",
            )

    def test_v3_teaches_lexical_role_and_attachment_ambiguity(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="A fresh prefix", target_lang="Chinese", committed_text="",
            num_candidates=20, prompt_version=SUFFIX_ICL_PROMPT_VERSION,
        )
        system = messages[0]["content"]
        self.assertEqual([example["kind"] for example in AMBIGUITY_ICL_EXAMPLES],
                         ["Word sense", "Grammatical role", "Phrase attachment",
                          "Object versus embedded subject across a relative clause",
                          "Main verb versus reduced passive relative clause"])
        self.assertIn("already observed words or relations could be translated", system)
        self.assertIn("Merely changing the later action, adjective, or intensity", system)
        for example in AMBIGUITY_ICL_EXAMPLES:
            with self.subTest(kind=example["kind"]):
                self.assertIn(f"Observed prefix: {example['prefix']}", system)
                response = f"Plausible\n1. {example['plausible']}\nContrastive\n1. {example['contrastive']}"
                self.assertIn(response, system)
                self.assertIn(example["explanation"], system)
                self.assertEqual(parse_grouped_future_output(response, 20),
                                 [(mode, example[mode]) for mode in ("plausible", "contrastive")])
                for mode in ("plausible", "contrastive"):
                    count = len(re.findall(r"[A-Za-z0-9']+", example[mode]))
                    self.assertGreaterEqual(count, 4)
                    self.assertLessEqual(count, 15)
                    self.assertFalse(example[mode].startswith(example["prefix"]))

    def test_v3_nested_and_passive_examples_preserve_the_shared_prefix(self) -> None:
        editor, soldiers = AMBIGUITY_ICL_EXAMPLES[-2:]
        self.assertEqual(editor["prefix"],
                         "The editor knew the author whom the reviewers, despite their reservations, praised")
        self.assertEqual(editor["plausible"], "from a conference they had attended together.")
        self.assertEqual(editor["contrastive"], "would refuse to revise the final chapter.")
        self.assertIn("In both readings, whom is the object of praised", editor["explanation"])
        self.assertEqual(soldiers["prefix"], "The soldiers warned about the ambush")
        self.assertEqual(soldiers["plausible"], "and advised the convoy to take another route.")
        self.assertEqual(soldiers["contrastive"], "were ordered to stay inside the camp overnight.")
        self.assertIn("equivalent to who were warned", soldiers["explanation"])

    def test_ambiguity_examples_do_not_change_legacy_prompt(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="A fresh prefix", target_lang="Chinese", committed_text="", num_candidates=20,
        )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertNotIn("Ambiguity example", prompt)
        for example in AMBIGUITY_ICL_EXAMPLES:
            self.assertNotIn(example["plausible"], prompt)


class GroupedFutureParserTest(unittest.TestCase):
    def test_short_groups_keep_labels_with_reset_or_global_numbering(self) -> None:
        for contrastive_index in (1, 3, 11):
            with self.subTest(index=contrastive_index):
                response = (
                    "Plausible\n1. could arrive before the storm.\n2. may leave during the night.\n"
                    f"Contrastive\n{contrastive_index}. might stay despite the warning.\n"
                )
                items = parse_grouped_future_output(response, 20)
                self.assertEqual([mode for mode, _ in items], ["plausible", "plausible", "contrastive"])
                self.assertEqual(len(items), 3)

    def test_either_group_can_be_empty(self) -> None:
        for empty_mode, filled_mode in (("Plausible", "Contrastive"), ("Contrastive", "Plausible")):
            response = f"{empty_mode}\nNone\n{filled_mode}\n1. could arrive before the storm."
            self.assertEqual(parse_grouped_future_output(response, 20),
                             [(filled_mode.lower(), "could arrive before the storm.")])

    def test_explicit_abstention_returns_no_candidates(self) -> None:
        self.assertEqual(parse_grouped_future_output("Plausible\nNone\nContrastive\nNone", 20), [])

    def test_common_heading_decoration_is_tolerated(self) -> None:
        response = "**Plausible:**\n1) could arrive before the storm.\n### Contrastive\nNone"
        self.assertEqual(parse_grouped_future_output(response, 20),
                         [("plausible", "could arrive before the storm.")])

    def test_malformed_or_truncated_output_is_not_silent_abstention(self) -> None:
        responses = [
            "", "1. no group heading here", "Plausible\nNone",
            "Plausible\nNone\nContrastive", "Plausible\nNone\nContrastive\nexplanation",
            "Plausible\nNone\n1. cannot mix empty and nonempty\nContrastive\nNone",
            "Plausible\n1. first candidate here\n1. duplicated number here\nContrastive\nNone",
            "Plausible\nNone\nPlausible\nNone\nContrastive\nNone",
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_grouped_future_output(response, 20)

    def test_benign_format_variations_are_tolerated(self) -> None:
        cases = {
            "Here are the suffixes:\n**Plausible:**\n1) could arrive before the storm.\n### Contrastive\n(none)":
                [("plausible", "could arrive before the storm.")],
            "Plausible\n1. None\nContrastive\n- might stay despite the warning.":
                [("contrastive", "might stay despite the warning.")],
            "Plausible\nContrastive\n1. might stay despite the warning.":
                [("contrastive", "might stay despite the warning.")],
            "<think>\nplanning\n</think>\nPlausible candidates\n(1) \"could arrive before the storm.\"\nContrastive candidates\nN/A":
                [("plausible", "could arrive before the storm.")],
            "Plausible\n1. **could arrive before the storm.**\nContrastive\nNone.":
                [("plausible", "could arrive before the storm.")],
            # Observed from Qwen3.8-27B on the prefixes "I", "A" and "Most sure and" (pilot 10345238).
            "Plausible\n1. hope that this will help everyone understand.\n\nContrast\n1. had never expected such a strong reaction.":
                [("plausible", "hope that this will help everyone understand."),
                 ("contrastive", "had never expected such a strong reaction.")],
            "Plausible\n1. of it, I proceeded to the next step.\nContrast\nNone":
                [("plausible", "of it, I proceeded to the next step.")],
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(parse_grouped_future_output(response, 20), expected)

    def test_trailing_empty_group_needs_none_unless_sampler_stopped(self) -> None:
        response = "Plausible\n1. could arrive before the storm.\nContrastive"
        self.assertEqual(parse_grouped_future_output(response, 20, finish_reason="stop"),
                         [("plausible", "could arrive before the storm.")])
        for finish_reason in (None, "length"):
            with self.subTest(finish_reason=finish_reason), self.assertRaises(ValueError):
                parse_grouped_future_output(response, 20, finish_reason=finish_reason)

    def test_length_truncated_response_is_rejected_even_with_content(self) -> None:
        response = "Plausible\n1. could arrive before the storm.\nContrastive\n1. because she wanted to"
        self.assertEqual(len(parse_grouped_future_output(response, 20, finish_reason="stop")), 2)
        with self.assertRaises(ValueError):
            parse_grouped_future_output(response, 20, finish_reason="length")

    def test_prose_inside_a_group_is_still_malformed(self) -> None:
        for response in (
            "Plausible\n1. could arrive before the storm.\nJoined check: fine.\nContrastive\nNone",
            "Plausible\nNone\nContrastive\n1. might stay.\nWhy: it resolves the ambiguity.",
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_grouped_future_output(response, 20, finish_reason="stop")

    def test_per_group_budget_is_enforced(self) -> None:
        response = "Plausible\n1. a valid first suffix\n2. a valid second suffix\n3. over budget\nContrastive\nNone"
        with self.assertRaises(ValueError):
            parse_grouped_future_output(response, 4)

    def test_bad_budgets_are_rejected(self) -> None:
        for budget in (0, -2, 3):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                parse_grouped_future_output("Plausible\nNone\nContrastive\nNone", budget)


class SampleGroupedFuturesTest(unittest.TestCase):
    GOOD = "Plausible\n1. could arrive before the storm.\nContrastive\nNone"
    BAD = "Plausible\n1. could arrive before the storm.\nContrastone\n1. might stay."

    def _run(self, responses, retries=2):
        events = []
        calls = []

        def request(attempt):
            calls.append(attempt)
            item = responses[attempt]
            if isinstance(item, Exception):
                raise item
            return item

        def record(event, attempt, raw_text, finish_reason, error):
            events.append((event, attempt, raw_text, error))

        return request, record, events, calls

    def test_retry_recovers_and_records_each_stage(self) -> None:
        request, record, events, calls = self._run([(self.BAD, "stop"), (self.GOOD, "stop")])
        items, raw = sample_grouped_futures(request, 20, 2, record)
        self.assertEqual(items, [("plausible", "could arrive before the storm.")])
        self.assertEqual(raw, self.GOOD)
        self.assertEqual(calls, [0, 1])
        self.assertEqual([(e[0], e[1]) for e in events], [("malformed", 0), ("recovered", 1)])
        self.assertEqual(events[0][2], self.BAD)

    def test_exhausted_retries_raise_after_recording(self) -> None:
        request, record, events, calls = self._run([(self.BAD, "stop")] * 3)
        with self.assertRaises(ValueError):
            sample_grouped_futures(request, 20, 2, record)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual([e[0] for e in events], ["malformed"] * 3 + ["exhausted"])

    def test_malformed_response_is_recorded_before_a_later_request_fails(self) -> None:
        request, record, events, calls = self._run([(self.BAD, "stop"), RuntimeError("endpoint down")])
        with self.assertRaises(RuntimeError):
            sample_grouped_futures(request, 20, 2, record)
        self.assertEqual([(e[0], e[2]) for e in events], [("malformed", self.BAD)])

    def test_first_good_response_records_nothing(self) -> None:
        request, record, events, calls = self._run([(self.GOOD, "stop")])
        sample_grouped_futures(request, 20, 2, record)
        self.assertEqual(events, [])
        self.assertEqual(calls, [0])


if __name__ == "__main__":
    unittest.main()
