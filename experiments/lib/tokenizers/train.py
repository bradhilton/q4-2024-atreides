import datasets
import tiktoken
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.processors import ByteLevel

fineweb: datasets.IterableDatasetDict = datasets.load_dataset(
    "HuggingFaceFW/fineweb", name="sample-10BT", streaming=True
)  # type: ignore

encoding = tiktoken.encoding_for_model("gpt-4o")

tokenizer = Tokenizer(models.BPE())

# Set up pre-tokenization and decoding similar to GPT tokenizers
tokenizer.pre_tokenizer = pre_tokenizers.Split(
    Regex(encoding._pat_str), behavior="merged_with_next"
)
tokenizer.decoder = decoders.ByteLevel()

# Set up post-processor
tokenizer.post_processor = ByteLevel(trim_offsets=False)

trainer = trainers.BpeTrainer(
    vocab_size=encoding.max_token_value + 1,
    special_tokens=list(encoding.special_tokens_set),
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tokenizer.train_from_iterator(
    (document["text"] for document in fineweb["train"].take(10_000)),
    trainer=trainer,
    length=10_000,
)

# Get the vocabulary from the gpt-4o encoding
gpt4o_vocab = set(encoding._mergeable_ranks.keys())

print(f"GPT-4o Vocabulary Length: ", len(gpt4o_vocab))

# Get the vocabulary from the newly trained tokenizer
new_vocab = set(key.encode() for key in tokenizer.get_vocab().keys())

print(f"Trained Vocabulary Length: ", len(gpt4o_vocab))

# Calculate similarity
common_tokens = gpt4o_vocab.intersection(new_vocab)
similarity = len(common_tokens) / len(gpt4o_vocab)

print(f"Vocabulary similarity: {similarity:.2%}")
