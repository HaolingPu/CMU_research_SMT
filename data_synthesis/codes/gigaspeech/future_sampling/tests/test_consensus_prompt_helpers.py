from pathlib import Path
import io
import json
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


class FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        self.messages = messages
        return [10, 20, 30] if kwargs["tokenize"] else "rendered-no-think"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class ConsensusPromptHelpersTest(unittest.TestCase):
    def make_args(self, *extra):
        argv = ['decoder', '--instruct-tokenizer-path', 'unused',
                '--instruct-api-base', 'http://unused', '--use-targeted-instruct-sampling',
                '--future-source-window-chunks', '1', *extra]
        with patch.object(sys, 'argv', argv):
            return decoder.parse_args()

    def test_production_defaults_leave_boundary_features_off(self):
        args = self.make_args()
        self.assertEqual(args.future_source_window_mode, 'fixed')
        self.assertEqual(args.future_join_mode, 'space')
        self.assertEqual(args.sentence_end_boundary_mode, 'literal')
        self.assertFalse(args.sentence_end_completion)
        self.assertIsNone(args.targeted_sampler_seed)
        self.assertEqual(args.targeted_sampler_context, 'source-and-target')
        self.assertEqual(args.sentence_end_punctuation, 'off')
        self.assertFalse(args.targeted_fail_on_api_error)
        self.assertEqual(args.targeted_prompt_version, decoder.PROMPT_VERSION)

    def test_source_only_prompt_and_seed_ignore_target(self):
        requests = []
        for committed in ['first-target', 'different-target']:
            tokenizer = FakeTokenizer()
            with patch.object(decoder, '_http_json', return_value={'choices': [{'text': ''}]}) as api:
                decoder._sample_coordinated_future_set(
                    tokenizer, 'I love music, and you', committed, 'Chinese', 4,
                    'http://localhost:1/v1', 'sampler', 1, 1, .98, 40,
                    sampler_seed=42, sampler_context='source-only', fail_on_api_error=True,
                )
            prompt_text = str(tokenizer.messages)
            self.assertNotIn(committed, prompt_text)
            self.assertNotIn('already committed', prompt_text)
            requests.append(api.call_args.kwargs['payload'])
        self.assertEqual(requests[0], requests[1])

    def test_pilot_does_not_hide_sampler_api_failure(self):
        with patch.object(decoder, '_http_json', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError):
                decoder._sample_coordinated_future_set(
                    FakeTokenizer(), 'I love music', '', 'Chinese', 4,
                    'http://localhost:1/v1', 'sampler', 1, 1, .98, 40,
                    fail_on_api_error=True,
                )

    def test_source_only_boundary_flow_and_punctuation(self):
        chunks = ['I love music,', 'and you', 'love basketball.', '', 'What next?']
        row = {'id': 'pilot', 'src_trajectory': repr(chunks),
               'src_text_full': repr(['I love music,', 'and you love basketball.', 'What next?']),
               'src_text': 'I love music, and you love basketball. What next?'}
        args = self.make_args('--future-source-window-mode', 'sentence-anchor',
                              '--targeted-sampler-context', 'source-only',
                              '--sentence-end-completion', '--sentence-end-boundary-mode', 'conservative',
                              '--sentence-end-punctuation', 'match-source')
        verbose = io.StringIO()
        with patch.object(decoder, 'sample_source_futures_targeted_prefill', return_value=([], [], [])) as sample:
            with patch.object(decoder, 'force_complete_translation', side_effect=['translation,', 'question.']) as final:
                result = decoder.run_one_utterance(row, args, [], FakeTokenizer(), verbose_log_file=verbose)
        self.assertEqual([c.kwargs['observed_source'] for c in sample.call_args_list],
                         ['I love music,', 'I love music, and you'])
        self.assertTrue(all(c.kwargs['sampler_context'] == 'source-only' for c in sample.call_args_list))
        self.assertEqual(result['target_trajectory'], ['', '', 'translation\u3002', '', 'question\uFF1F'])
        self.assertEqual(final.call_args.kwargs['committed_text'], 'translation\u3002')
        self.assertEqual(result['boundary_audit'][0]['raw_delta'], 'translation,')
        self.assertEqual(final.call_args.kwargs['full_source'], row['src_text'])
        audit = result['sampling_audit']
        self.assertEqual([event['status'] for event in audit], ['invoked', 'invoked', 'skipped', 'skipped', 'skipped'])
        self.assertEqual([event['reason'] for event in audit[2:]], ['source_sentence_end', 'no_new_source', 'final_chunk'])
        self.assertEqual([event['input_source_prefix'] for event in audit[:2]],
                         [call.kwargs['observed_source'] for call in sample.call_args_list])
        self.assertTrue(all(event['input_source_prefix'] is None for event in audit[2:]))
        self.assertTrue(all(event['committed_target_context'] == '' for event in audit))
        logged = [json.loads(line.split(' ', 1)[1]) for line in verbose.getvalue().splitlines()
                  if line.startswith('[FutureSampling] ')]
        self.assertEqual(logged, audit)

    def test_input_log_exists_before_sampler_failure(self):
        row = {'id': 'failure', 'src_trajectory': repr(['Hello', 'world.']),
               'src_text_full': repr(['Hello world.']), 'src_text': 'Hello world.'}
        verbose = io.StringIO()
        with patch.object(decoder, 'sample_source_futures_targeted_prefill', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError):
                decoder.run_one_utterance(row, self.make_args(), [], FakeTokenizer(), verbose_log_file=verbose)
        event = json.loads(next(line.split(' ', 1)[1] for line in verbose.getvalue().splitlines()
                                if line.startswith('[FutureSampling] ')))
        self.assertEqual(event['input_source_prefix'], 'Hello')
        self.assertEqual(event['status'], 'invoked')

    def test_sentence_anchor_changes_only_sampler_prefix(self):
        chunks = ['First ended.', 'This giant was furious,', 'and words were', 'just coming.']
        row = {'id': 'test', 'src_trajectory': repr(chunks),
               'src_text_full': repr(['First ended.', 'This giant was furious,',
                                      'and words were just coming.']), 'src_text': ' '.join(chunks)}
        prefixes = {}
        for mode in ['fixed', 'sentence-anchor']:
            with patch.object(decoder, 'sample_source_futures_targeted_prefill', return_value=([], [], [])) as sample:
                with patch.object(decoder, 'force_complete_translation', return_value='') as final:
                    result = decoder.run_one_utterance(
                        row, self.make_args('--future-source-window-mode', mode), [], FakeTokenizer()
                    )
            prefixes[mode] = sample.call_args_list[2].kwargs['observed_source'].strip()
            self.assertEqual(final.call_args.kwargs['full_source'], ' '.join(chunks))
            self.assertEqual(result['actions'], ['READ'] * 4)
        self.assertEqual(prefixes['fixed'], 'and words were')
        self.assertEqual(prefixes['sentence-anchor'], 'This giant was furious, and words were')

    def test_conservative_completion_defers_abbreviation_and_empty_chunk(self):
        chunks = ['I spoke to Dr.', 'Smith.', '', 'Then we left.']
        row = {'id': 'test', 'src_trajectory': repr(chunks),
               'src_text_full': repr(['I spoke to Dr. Smith.', 'Then we left.']),
               'src_text': 'I spoke to Dr. Smith. Then we left.'}
        args = self.make_args('--sentence-end-completion', '--sentence-end-boundary-mode', 'conservative')
        with patch.object(decoder, 'sample_source_futures_targeted_prefill', return_value=([], [], [])):
            with patch.object(decoder, 'force_complete_translation', return_value='') as final:
                decoder.run_one_utterance(row, args, [], FakeTokenizer())
        self.assertEqual([call.kwargs['full_source'] for call in final.call_args_list],
                         ['I spoke to Dr. Smith.', 'I spoke to Dr. Smith. Then we left.'])

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

    def test_future_cleanup_strips_repeated_prefix_case_insensitively(self) -> None:
        self.assertEqual(
            decoder.clean_future_text("And these", "and these are the main challenges"),
            "are the main challenges",
        )
        self.assertEqual(
            decoder.clean_future_text(
                "A sense of literary honesty compels the editor",
                "EDITOR is merely a curator",
            ),
            "is merely a curator",
        )
        self.assertEqual(decoder.clean_future_text("The areas", "AREAS remain open"), "remain open")
        self.assertEqual(decoder.clean_future_text("The areas are", "area remains open"), "area remains open")

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

    def test_v3_short_response_does_not_fabricate_missing_candidates(self):
        tokenizer = FakeTokenizer()
        response = {"choices": [{"text": json.dumps({
            "plausible": ["could arrive before the storm.", "may leave during the night."],
            "contrastive": ["might stay despite the warning."],
        }), "finish_reason": "stop"}]}
        with patch.object(decoder, '_http_json', return_value=response) as api:
            futures, infos, audit = decoder._sample_coordinated_future_set(
                tokenizer, 'The guests', 'never send this target', 'Chinese', 20,
                'http://localhost:1/v1', 'sampler', 1, 1, .98, 40,
                sampler_context='source-only', prompt_version=decoder.SUFFIX_ICL_PROMPT_VERSION,
            )
        self.assertEqual(len(futures), 3)
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(item['accepted'] for item in audit))
        self.assertEqual([item['mode'] for item in audit], ['plausible', 'plausible', 'contrastive'])
        self.assertTrue(all(item['prompt_version'] == decoder.SUFFIX_ICL_PROMPT_VERSION for item in infos))
        self.assertNotIn('never send this target', str(tokenizer.messages))
        self.assertIn('Never pad a list', tokenizer.messages[0]['content'])
        schema = api.call_args.kwargs['payload']['structured_outputs']['json']
        self.assertEqual(schema['required'], ['plausible', 'contrastive'])
        self.assertEqual(schema['properties']['plausible']['maxItems'], 10)

    def test_v3_does_not_disguise_malformed_response_as_fewer_good_candidates(self):
        with patch.object(decoder, '_http_json', return_value={'choices': [{'text': 'unstructured text'}]}):
            with self.assertRaisesRegex(ValueError, 'Invalid future_set_v3_suffix_icl response from sampler'):
                decoder._sample_coordinated_future_set(
                    FakeTokenizer(), 'The guests', '', 'Chinese', 20,
                    'http://localhost:1/v1', 'sampler', 1, 1, .98, 40,
                    prompt_version=decoder.SUFFIX_ICL_PROMPT_VERSION,
                )

    def test_v3_version_is_forwarded_to_both_samplers(self):
        with patch.object(decoder, '_sample_coordinated_future_set', return_value=([], [], [])) as sample:
            result = decoder.sample_source_futures_targeted_prefill(
                FakeTokenizer(), 'The guests', '', 'Chinese', 20,
                'http://localhost:1/v1', 'first-model', 1,
                sampler2_tokenizer=FakeTokenizer(), sampler2_api_base='http://localhost:2/v1',
                sampler2_api_model='second-model', return_audit=True,
                prompt_version=decoder.SUFFIX_ICL_PROMPT_VERSION,
            )
        self.assertEqual(result, ([], [], []))
        self.assertEqual(sample.call_count, 2)
        self.assertTrue(all(c.kwargs['prompt_version'] == decoder.SUFFIX_ICL_PROMPT_VERSION
                            for c in sample.call_args_list))

    def test_v3_icl_uses_each_models_template_and_one_completion_request(self):
        tokenizers = [FakeTokenizer(), FakeTokenizer()]
        response = {'choices': [{'text': '{"plausible": [], "contrastive": []}', 'finish_reason': 'stop'}]}
        with patch.object(decoder, '_http_json', return_value=response) as api:
            decoder.sample_source_futures_targeted_prefill(
                tokenizers[0], 'This giant was even more furious than the first',
                'private committed target', 'Chinese', 20, 'http://gemma/v1', 'gemma4-sampler', 120,
                sampler2_tokenizer=tokenizers[1], sampler2_api_base='http://qwen/v1',
                sampler2_api_model='qwen38-sampler', sampler_seed=1015,
                sampler_context='source-only', prompt_version=decoder.SUFFIX_ICL_PROMPT_VERSION,
            )
        self.assertEqual(api.call_count, 2)
        self.assertEqual([call.args[0] for call in api.call_args_list],
                         ['http://gemma/v1/completions', 'http://qwen/v1/completions'])
        self.assertEqual(tokenizers[0].messages, tokenizers[1].messages)
        for tokenizer, call in zip(tokenizers, api.call_args_list):
            self.assertEqual(tokenizer.kwargs,
                             {'add_generation_prompt': True, 'tokenize': False, 'enable_thinking': False})
            self.assertIn('Ambiguity example 1 (Word sense)', tokenizer.messages[0]['content'])
            self.assertNotIn('private committed target', str(tokenizer.messages))
            payload = call.kwargs['payload']
            self.assertEqual(payload['prompt'], 'rendered-no-think')
            self.assertEqual(payload['n'], 1)
            self.assertEqual(payload['max_tokens'], 640)
            self.assertEqual(payload['temperature'], 1.0)
            self.assertEqual(payload['top_p'], .98)
            self.assertEqual(payload['presence_penalty'], .15)
            self.assertIn('seed', payload)

    def test_v3_zero_or_too_few_candidates_read_without_probe(self):
        responses = [
            '{"plausible": [], "contrastive": []}',
            '{"plausible": ["could arrive before the storm."], "contrastive": []}',
        ]
        for text in responses:
            with self.subTest(response=text):
                row = {'id': 'short-list', 'src_trajectory': repr(['The guests', 'arrived.']),
                       'src_text_full': repr(['The guests arrived.']), 'src_text': 'The guests arrived.'}
                args = self.make_args('--targeted-prompt-version', decoder.SUFFIX_ICL_PROMPT_VERSION,
                                      '--targeted-sampler-context', 'source-only')
                verbose = io.StringIO()
                with patch.object(decoder, '_http_json', return_value={'choices': [{'text': text}]}):
                    with patch.object(decoder, 'extend_pending_tokens') as probe:
                        with patch.object(decoder, 'force_complete_translation', return_value=''):
                            result = decoder.run_one_utterance(
                                row, args, [], FakeTokenizer(), sampler_tokenizer=FakeTokenizer(),
                                verbose_log_file=verbose,
                            )
                probe.assert_not_called()
                self.assertEqual(result['actions'][0], 'READ')
                self.assertEqual(result['target_trajectory'][0], '')
                self.assertEqual(result['decoder_settings']['targeted_prompt_version'],
                                 decoder.SUFFIX_ICL_PROMPT_VERSION)
                self.assertEqual(result['sampling_audit'][0]['prompt_version'], decoder.SUFFIX_ICL_PROMPT_VERSION)
                self.assertIn('# targeted_prompt_version: future_set_v3_suffix_icl', verbose.getvalue())


class ProbeAndVoteChangesTest(unittest.TestCase):
    def test_heard_guessed_probe_prompt_puts_shared_text_before_the_future(self):
        tokenizer = FakeTokenizer()
        decoder.build_translation_probe_prompt_prefix_token_ids(
            tokenizer, "And the fork flew into a", True, guessed_continuation="dozen pieces on the floor.")
        content = tokenizer.messages[0]["content"]
        self.assertIn("[HEARD]\nAnd the fork flew into a", content)
        self.assertIn("Translate only what was heard", content)
        self.assertTrue(content.endswith("[POSSIBLE CONTINUATION]\ndozen pieces on the floor."))
        self.assertLess(content.index("[IMPORTANT]"), content.index("[POSSIBLE CONTINUATION]"))
        decoder.build_translation_probe_prompt_prefix_token_ids(tokenizer, "And the fork flew into a", True)
        self.assertNotIn("[HEARD]", tokenizer.messages[0]["content"])

    def test_absolute_voter_floor_blocks_thin_unanimity(self):
        distributions = [{7: 0.9, 8: 0.1}] * 4
        token, meta = decoder.choose_consensus_token(distributions, min_voters_ratio=1.0)
        self.assertEqual(token, 7)
        token, meta = decoder.choose_consensus_token(distributions, min_voters_ratio=1.0, min_voters_abs=10)
        self.assertIsNone(token)
        self.assertEqual(meta["min_voters"], 10)


if __name__ == "__main__":
    unittest.main()
