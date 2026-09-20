"""Explicit, collision-free FAST vocabulary extension for a Gemma tokenizer."""

import numpy as np


class FASTCodec:
    """Map a supplied pretrained FAST processor's codes into new Gemma tokens.

    The paper does not disclose its vocabulary mapping. This implementation adds
    distinct tokens instead of overwriting existing language tokens. Save the
    extended text tokenizer alongside the model. FAST operates on normalized
    actions; padded timesteps repeat the final valid action before DCT encoding.
    """

    def __init__(self, processor, text_tokenizer, *, codebook_size: int):
        if isinstance(codebook_size, bool) or not isinstance(codebook_size, int) or codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer")
        self.processor = processor
        self.codebook_size = codebook_size
        tokens = [f"<|pi07_action_{index}|>" for index in range(codebook_size)]
        text_tokenizer.add_tokens(tokens, special_tokens=True)
        self.token_ids = np.asarray(text_tokenizer.convert_tokens_to_ids(tokens), dtype=np.int64)
        if len(np.unique(self.token_ids)) != codebook_size:
            raise ValueError("FAST vocabulary must map to distinct text tokens")
        self.eos_id = text_tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("Text tokenizer must provide an EOS token")

    def __call__(self, actions, action_mask):
        actions, mask = np.asarray(actions), np.asarray(action_mask)
        if actions.ndim != 2 or actions.shape != mask.shape or mask.dtype != bool:
            raise ValueError("FAST actions and boolean mask must have shape [horizon,dimension]")
        if not np.issubdtype(actions.dtype, np.number) or np.iscomplexobj(actions) or not np.isfinite(actions).all():
            raise ValueError("FAST actions must be finite real numbers")
        valid_rows = np.flatnonzero(mask.any(-1))
        if not valid_rows.size:
            raise ValueError("FAST training needs at least one valid action")
        if not np.array_equal(valid_rows, np.arange(valid_rows[-1] + 1)):
            raise ValueError("FAST timestep padding must be a suffix")
        prepared = np.where(mask, actions, 0).copy()
        prepared[valid_rows[-1] + 1 :] = prepared[valid_rows[-1]]
        codes = np.asarray(self.processor(prepared[None])[0])
        if codes.ndim != 1 or not codes.size or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("FAST processor must return integer codes")
        if np.any(codes < 0) or np.any(codes >= self.codebook_size):
            raise ValueError("FAST processor returned a code outside its declared codebook")
        return np.concatenate((self.token_ids[codes], [self.eos_id]))
