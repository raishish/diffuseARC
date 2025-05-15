from transformers import PreTrainedTokenizerFast


def create_arc_tokenizer(config: dict) -> PreTrainedTokenizerFast:
    """Create a simple tokenizer for ARC tasks with digits and special tokens"""
    vocab = {str(i): i for i in range(10)}  # Digits 0-9

    # Add special tokens
    special_tokens = {
        "<pad>": 10,
        "<s>": 11,
        "</s>": 12,
        "<start_grid>": 13,
        "<end_grid>": 14,
        "<start_row>": 15,
        "<end_row>": 16,
        "<mask>": 17  # For masked diffusion
    }

    vocab.update(special_tokens)
    inv_vocab = {v: k for k, v in vocab.items()}

    # Create tokenizer (simplified approach)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=None,  # Would need to create a tokenizer object
        vocab_size=len(vocab),
        model_max_length=32768,  # Qwen's extended context length
        padding_side="right",
        truncation_side="right",
    )

    # Set vocabulary and special tokens
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "mask_token": "<mask>",
        "additional_special_tokens": ["<start_grid>", "<end_grid>", "<start_row>", "<end_row>"]
    })

    return tokenizer
