"""A stand-in for the Gemma 4 GGUF in the wrapper tests: its chat template's
shape and its control tokens, so that a prompt can be read back."""

import re
from unittest.mock import MagicMock

# As Gemma 4's: content is trimmed; thinking on puts <|think|> in the system turn
# and leaves the model's turn open, off it closes an empty thought.
TEMPLATE = (
    "{{ bos_token }}<|turn>system\n{% if enable_thinking %}<|think|>\n{% endif %}"
    "{{ messages[0]['content'] | trim }}<turn|>\n"
    "<|turn>user\n{{ messages[1]['content'] | trim }}<turn|>\n<|turn>model\n"
    "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}"
)
CONTROL = {
    "<eos>": 1,
    "<bos>": 2,
    "<|think|>": 98,
    "<|channel>": 100,
    "<channel|>": 101,
    "<|turn>": 105,
    "<turn|>": 106,
}
_NAMES = {token: name for name, token in CONTROL.items()}
_MARKS = re.compile("(" + "|".join(map(re.escape, CONTROL)) + ")")


def tokenize(text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
    """One token a character; a control mark is one token only when special."""
    tokens = []
    for part in _MARKS.split(text.decode()) if special else [text.decode()]:
        tokens += (
            [CONTROL[part]]
            if special and part in CONTROL
            else [ord(c) + 1000 for c in part]
        )
    return tokens


def detokenize(tokens: list[int], special: bool = False) -> bytes:
    return "".join(
        (_NAMES[t] if special else "") if t in _NAMES else chr(t - 1000) for t in tokens
    ).encode()


def read(tokens: list[int]) -> str:
    """The prompt as text, control tokens by their names."""
    return detokenize(tokens, special=True).decode()


def controls(tokens: list[int]) -> list[str]:
    """The prompt's control tokens alone, in order."""
    return [_NAMES[t] for t in tokens if t in _NAMES]


def gemma(text="", finish_reason="stop", n_ctx=100_000, usage=None) -> MagicMock:
    model = MagicMock()
    model.metadata = {"tokenizer.chat_template": TEMPLATE}
    model.tokenize.side_effect = tokenize
    model.detokenize.side_effect = detokenize
    model.token_bos.return_value = CONTROL["<bos>"]
    model.token_eos.return_value = CONTROL["<eos>"]
    model.n_ctx.return_value = n_ctx
    model.create_completion.return_value = {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        **({"usage": usage} if usage else {}),
    }
    return model
