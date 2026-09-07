from pathlib import Path
import json
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
    grouped_future_schema,
    parse_grouped_future_output,
    sample_grouped_futures,
    structured_output_extras,
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
        self.assertIn('{"plausible": ["<suffix>", ...], "contrastive": ["<suffix>", ...]}', prompt)
        self.assertIn("Use an empty list for a group with no valid candidate", prompt)
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
                response = json.dumps({"plausible": [example["plausible"]], "contrastive": [example["contrastive"]]})
                self.assertIn(response, system)
                self.assertIn(example["explanation"], system)
                self.assertEqual(parse_grouped_future_output(response, 20),
                                 [(mode, example[mode], "") for mode in ("plausible", "contrastive")])
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
    def test_schema_caps_each_group_and_forbids_extra_keys(self) -> None:
        schema = grouped_future_schema(20)
        self.assertEqual(schema["required"], ["plausible", "contrastive"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["contrastive"]["maxItems"], 10)
        self.assertEqual(structured_output_extras(4)["structured_outputs"]["json"]["properties"]["plausible"]["maxItems"], 2)
        for budget in (0, -2, 3):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                grouped_future_schema(budget)

    def test_groups_keep_labels_and_order(self) -> None:
        response = json.dumps({"contrastive": ["might stay despite the warning."],
                               "plausible": ["could arrive before the storm.", " may leave during the night. "]})
        self.assertEqual(parse_grouped_future_output(response, 20, "stop"), [
            ("plausible", "could arrive before the storm.", ""), ("plausible", "may leave during the night.", ""),
            ("contrastive", "might stay despite the warning.", ""),
        ])

    def test_empty_groups_are_explicit_abstention(self) -> None:
        self.assertEqual(parse_grouped_future_output('{"plausible": [], "contrastive": []}', 20, "stop"), [])
        self.assertEqual(parse_grouped_future_output('{"plausible": ["could arrive before the storm."], "contrastive": []}', 20, "stop"),
                         [("plausible", "could arrive before the storm.", "")])

    def test_malformed_or_truncated_output_is_never_silent_abstention(self) -> None:
        responses = {
            "": "stop", "unstructured text": "stop", '{"plausible": []}': "stop",
            '{"plausible": [], "contrastive": [], "extra": []}': "stop",
            '{"plausible": ["ok suffix here."], "contrastive": [""]}': "stop",
            '{"plausible": [1], "contrastive": []}': "stop",
            '{"plausible": "not a list", "contrastive": []}': "stop",
            '{"plausible": ["a", "b", "c"], "contrastive": []}': "stop",  # over the budget of 2 per group
            '{"plausible": ["could arrive before the storm."], "contrastive": ["because she wanted to"]}': "length",
        }
        for response, finish_reason in responses.items():
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_grouped_future_output(response, 4, finish_reason)

    def test_contrastive_notes_schema_and_parse(self) -> None:
        schema = grouped_future_schema(20, contrastive_notes=True)
        item = schema["properties"]["contrastive"]["items"]
        self.assertEqual(item["required"], ["suffix", "resolves"])
        self.assertEqual(schema["properties"]["plausible"]["items"]["type"], "string")
        response = json.dumps({"plausible": ["to withdraw some cash."],
                               "contrastive": [{"suffix": "of the river to watch ducks.", "resolves": "bank = river edge"},
                                               "of the river, plain string still accepted."]})
        self.assertEqual(parse_grouped_future_output(response, 20, "stop"), [
            ("plausible", "to withdraw some cash.", ""),
            ("contrastive", "of the river to watch ducks.", "bank = river edge"),
            ("contrastive", "of the river, plain string still accepted.", ""),
        ])
        for bad in ('{"plausible": [], "contrastive": [{"suffix": "x y z"}]}',
                    '{"plausible": [], "contrastive": [{"suffix": "x y z", "resolves": 3}]}',
                    '{"plausible": [{"suffix": "x", "resolves": "y"}], "contrastive": []}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_grouped_future_output(bad, 20, "stop")

    def test_contrastive_notes_prompt_shows_object_form(self) -> None:
        messages = build_coordinated_future_messages(
            observed_source="The visitors stopped by the bank", target_lang="Chinese", committed_text="",
            num_candidates=20, prompt_version=SUFFIX_ICL_PROMPT_VERSION, contrastive_notes=True,
        )
        system, user = (m["content"] for m in messages)
        self.assertIn('"resolves": "bank = river edge, not the financial institution"', system)
        self.assertIn('"resolves": "<reading of an already observed word', user)
        self.assertIn("must name a different reading", user)
        plain = build_coordinated_future_messages(
            observed_source="The visitors stopped by the bank", target_lang="Chinese", committed_text="",
            num_candidates=20, prompt_version=SUFFIX_ICL_PROMPT_VERSION,
        )
        self.assertNotIn("resolves", plain[1]["content"])

    def test_bad_budgets_are_rejected(self) -> None:
        for budget in (0, -2, 3):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                parse_grouped_future_output('{"plausible": [], "contrastive": []}', budget)


class SampleGroupedFuturesTest(unittest.TestCase):
    GOOD = '{"plausible": ["could arrive before the storm."], "contrastive": []}'
    BAD = '{"plausible": ["could arrive before the storm."], "contrast": ["might stay."]}'

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
        self.assertEqual(items, [("plausible", "could arrive before the storm.", "")])
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
