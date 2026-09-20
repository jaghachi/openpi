"""FAST vocabulary/masking tests with a local real tokenizer and fake codec.

The fake processor checks integration contracts only; these tests do not claim
to reproduce FAST's learned DCT/tokenization behavior.
"""

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from openpi.pi07.tokenization import FASTCodec


def text_tokenizer():
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "fold": 4}, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )


class Processor:
    def __init__(self, codes=(0, 3, 1)):
        self.codes = np.asarray(codes)
        self.actions = None

    def __call__(self, actions):
        self.actions = actions.copy()
        return [self.codes]


def test_fast_tokens_are_disjoint_stable_and_preserve_original_vocabulary():
    tokenizer = text_tokenizer()
    original = tokenizer.get_vocab().copy()
    codec = FASTCodec(Processor(), tokenizer, codebook_size=4)
    encoded = codec(np.ones((5, 2)), np.ones((5, 2), dtype=bool))
    np.testing.assert_array_equal(encoded, [5, 8, 6, 2])
    assert {name: tokenizer.get_vocab()[name] for name in original} == original
    assert len(tokenizer) == 9
    reloaded_codec = FASTCodec(Processor(), tokenizer, codebook_size=4)
    np.testing.assert_array_equal(reloaded_codec.token_ids, codec.token_ids)
    assert len(tokenizer) == 9


def test_fast_padding_repeats_last_valid_action_and_masks_coordinates():
    processor = Processor()
    codec = FASTCodec(processor, text_tokenizer(), codebook_size=4)
    actions = np.arange(12, dtype=np.float32).reshape(4, 3)
    mask = np.array([[True, True, False], [True, True, False], [False] * 3, [False] * 3])
    codec(actions, mask)
    np.testing.assert_array_equal(processor.actions, [[[0, 1, 0], [3, 4, 0], [3, 4, 0], [3, 4, 0]]])
    np.testing.assert_array_equal(actions, np.arange(12).reshape(4, 3))


@pytest.mark.parametrize("codes", [[-1], [4], [0.5], [[0, 1]], np.array([], dtype=np.int64)])
def test_fast_rejects_invalid_processor_codes(codes):
    codec = FASTCodec(Processor(codes), text_tokenizer(), codebook_size=4)
    with pytest.raises(ValueError, match=r"integer codes|outside"):
        codec(np.ones((3, 2)), np.ones((3, 2), dtype=bool))


def test_fast_rejects_empty_supervision_holes_and_bad_mask_type():
    codec = FASTCodec(Processor(), text_tokenizer(), codebook_size=4)
    actions = np.ones((3, 2))
    with pytest.raises(ValueError, match="at least one valid"):
        codec(actions, np.zeros((3, 2), dtype=bool))
    with pytest.raises(ValueError, match="suffix"):
        codec(actions, np.array([[True, True], [False, False], [True, True]]))
    with pytest.raises(ValueError, match="boolean mask"):
        codec(actions, np.ones((3, 2), dtype=int))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1 + 2j, "bad"])
def test_fast_rejects_nonfinite_or_nonreal_actions(value):
    codec = FASTCodec(Processor(), text_tokenizer(), codebook_size=4)
    with pytest.raises(ValueError, match="finite real"):
        codec(np.full((3, 2), value), np.ones((3, 2), dtype=bool))


@pytest.mark.parametrize("size", [0, -1, True, 3.5])
def test_fast_rejects_invalid_codebook_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        FASTCodec(Processor(), text_tokenizer(), codebook_size=size)
