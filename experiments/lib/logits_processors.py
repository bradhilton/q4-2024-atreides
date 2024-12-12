def random_logits_processor(past_tokens, logits):
    return logits / 1_000
