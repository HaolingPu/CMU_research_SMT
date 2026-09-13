"""ms-swift loss-scale plugin: supervise the end token of EVERY assistant turn.

Loaded into the trainer with `--external_plugins <this file> --loss_scale
assistant_turn_end` (finetune/recipes/omni_sft_recipe.sh sets both from
OMNI_LOSS_SCALE). Nothing here runs outside the ms-swift container.

WHY IT EXISTS
-------------
ms-swift's default loss scale labels a response's own tokens and the FINAL
turn's end token, and nothing else. Between turns the template's `chat_sep`
(`<|im_end|>\\n` for the Qwen templates) is tagged ContextType.OTHER, scale 0,
and an EMPTY response tokenizes to nothing -- so an intermediate empty
assistant turn contributes no supervised token at all (a non-empty one gets its
`<|im_end|>` back through `_add_dynamic_eos`, which relabels a suffix that
follows a supervised token). Measured on this repo's manifests: the 55% empty
"wait" turns of the m=1 corpus carry no loss, and a checkpoint trained on them
emits the empty string on 15% of its calls (finetune/README.md, "The
wait-gated thinker"). Intermediate empty turns therefore provide no direct
supervision for the stop decision. Labeling their end tokens supplies that
missing target.

WHAT IT CHANGES
---------------
Exactly one thing: the `chat_sep` after an assistant turn gets its `<|im_end|>`
labelled (scale 1) and its trailing newline left unlabelled (scale 0). Responses
and the final suffix keep the default treatment, every prompt/system/user piece
stays at 0. For a non-empty turn this labels the token `_add_dynamic_eos` would
have labelled anyway; for an empty turn it is the only supervised token, and it
is the one that says "nothing more on this chunk". The model is otherwise
trained as before, so a run with this plugin differs from one without it only
in those positions -- and `args.json` records both flags, so the provenance
hash tells the two apart.

The match is on the literal separator string, not on a template name: a
template whose chat_sep is something else gets the default behaviour and the
dump in finetune/scripts/dump_labels.py shows it (no supervised end tokens on
empty turns), which is the check to run before training with a new template.
"""
from swift.llm.template.utils import ContextType
from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map

NAME = "assistant_turn_end"
# The Qwen chat templates' separator between rounds: the assistant's end token
# and the newline before the next `<|im_start|>`.
CHAT_SEP = "<|im_end|>\n"
END_TOKEN = "<|im_end|>"


class AssistantTurnEndLossScale(LossScale):
    """The default loss scale, plus a label on every assistant turn's end token."""

    def get_loss_scale(self, context, context_type, is_last_round, **kwargs):
        if context_type == ContextType.OTHER and context == CHAT_SEP:
            return [END_TOKEN, CHAT_SEP[len(END_TOKEN):]], [1.0, 0.0]
        return super().get_loss_scale(context, context_type, is_last_round, **kwargs)


AssistantTurnEndLossScale.name = NAME
loss_scale_map[NAME] = AssistantTurnEndLossScale
