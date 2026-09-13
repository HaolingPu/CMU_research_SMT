"""ms-swift loss-scale plugins: assistant turn ends supervised at a weight BELOW 1.

Loaded with `--external_plugins <this file> --loss_scale <name>` (the recipe maps the
names below to this file). Two families, each at a fixed set of weights:

    empty_turn_end_w<w>   the <|im_end|> that closes an EMPTY assistant turn gets loss
                          weight w; the end token after a non-empty turn, and every
                          response token, keep weight 1 (assistant_turn_end's rule)
    turn_end_w<w>         EVERY assistant <|im_end|> gets weight w, empty or not;
                          response tokens keep weight 1

    w in {0.1, 0.2, 0.25, 0.5, 0.75}

WHY
---
assistant_turn_end (weight 1 on every turn end) fixed what the default loss lost --
an empty turn had no supervised token -- but the checkpoints trained with it wait more
than their data: 70-75% of policy calls answered with the empty string against 55%
empty turns in the corpus, 13-15-character deltas against 9. The end token that closes
an empty turn is the token that says "nothing on this chunk"; a weight below 1 on it
lowers that pressure without touching what the model says when it speaks. The second
family also lowers the stop decision after content, for the comparison.

HOW
---
The default LossScale sees one context piece at a time and cannot know whether the
separator it is looking at follows an empty response, so this one overrides __call__
to remember the last RESPONSE piece. The separator between rounds is `<|im_end|>\\n`
(ContextType.OTHER); the final turn closes with `<|im_end|>` alone (ContextType.SUFFIX).
Both are split so the newline stays at 0. `is_binary = False` makes the template emit
a per-token `loss_scale` tensor next to the labels (a token with scale > 0 is labelled),
and ms-swift's Megatron SFT trainer multiplies the per-token losses by it
(swift/megatron/trainers/trainer.py, loss_func). The packing collator carries the key.
Everything a run used is in args.json (`loss_scale` = the name), so two weights are two
provenance configurations.
"""
from swift.llm.template.utils import ContextType
from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map

WEIGHTS = (0.1, 0.2, 0.25, 0.5, 0.75)
CHAT_SEP = "<|im_end|>\n"
END_TOKEN = "<|im_end|>"


class WeightedTurnEndLossScale(LossScale):
    """Turn-end loss weight `weight`, on empty turns only or on every turn."""

    is_binary = False
    weight = 1.0
    empty_only = True

    def __call__(self, context_list, context_types, messages, **kwargs):
        res_context, res_scale = [], []
        last_response = None
        for context, context_type in zip(context_list, context_types):
            if context_type == ContextType.RESPONSE:
                last_response = context if isinstance(context, str) else None
            pieces, scales = self._pieces(context, context_type, last_response)
            res_context += pieces
            res_scale += scales
        return res_context, res_scale

    def _pieces(self, context, context_type, last_response):
        if context_type == ContextType.RESPONSE:
            return [context], [1.0]
        if isinstance(context, str) and context_type == ContextType.OTHER and context == CHAT_SEP:
            return [END_TOKEN, CHAT_SEP[len(END_TOKEN):]], [self._end_weight(last_response), 0.0]
        if isinstance(context, str) and context_type == ContextType.SUFFIX and context == END_TOKEN:
            return [context], [self._end_weight(last_response)]
        if context_type == ContextType.SUFFIX:
            return [context], [1.0]
        return [context], [0.0]

    def _end_weight(self, last_response):
        if self.empty_only and last_response:
            return 1.0
        return float(self.weight)


def _register(family, empty_only):
    for w in WEIGHTS:
        name = f"{family}_w{w:g}"
        cls = type(f"{family}_w{str(w).replace('.', '_')}", (WeightedTurnEndLossScale,),
                   {"weight": w, "empty_only": empty_only, "name": name})
        loss_scale_map[name] = cls


_register("empty_turn_end", True)
_register("turn_end", False)
NAMES = tuple(f"{family}_w{w:g}" for family in ("empty_turn_end", "turn_end") for w in WEIGHTS)
