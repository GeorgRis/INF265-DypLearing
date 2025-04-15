import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class QADataset(Dataset):
    def __init__(
        self, config, tokenizer, 
    ):
        self.dataset = load_dataset(config.dataset)[config.split]
        n_subset = int(config.model_train_fraction * len(self.dataset))
        self.dataset= self.dataset.select(range(n_subset))
        print(
            f"Loaded dataset of size {len(self.dataset)} with columns {self.dataset.column_names}"
        )

        self.tokenizer = tokenizer
        self.max_length = config.max_len

        # Special token IDs (you can use these IDs in the __getitem__ method)
        self.pad_id = self.tokenizer.token_to_id(config.pad_token)
        self.sep_id = self.tokenizer.token_to_id(config.sep_token)
        self.end_id = self.tokenizer.token_to_id(config.end_token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question, answer = self.dataset[idx]["question"], self.dataset[idx]["answer"]

        # 1. Tokenize
        question_ids = self.tokenizer.encode(question).ids
        answer_ids = self.tokenizer.encode(answer).ids

        # 2. Combine into full sequence with special tokens
        full_sequence = question_ids + [self.sep_id] + answer_ids + [self.end_id]

        # 3. Truncate/pad to max_length + 1
        padded = full_sequence[:self.max_length + 1]  # Truncate if too long
        if len(padded) < self.max_length + 1:
            padded += [self.pad_id] * (self.max_length + 1 - len(padded))

        padded = torch.tensor(padded)

        # 4. Create source and target sequences
        source_sequence = padded[:-1]
        target_sequence = padded[1:].clone()
        target_sequence[source_sequence == self.pad_id] = -100  # Ignore loss on PAD

        # 5. Padding mask (True where PAD token is)
        key_padding_mask = (source_sequence == self.pad_id)

        return {
            "source_sequence": source_sequence,
            "target_sequence": target_sequence,
            "key_padding_mask": key_padding_mask,
        }


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from datasets import load_dataset

    # Sanity check the dataset class
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    idx = 1
    config.max_len = 64 # For testing purposes
    dataset = QADataset(config, tokenizer)

    source, target, key_padding_mask = dataset[idx].values()

    print("Source sequence shape:", source.shape)
    print("Target sequence shape:", target.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    print("Source sequence:", source)
    print("Target sequence:", target)
    print("Key padding mask:", key_padding_mask)

    decoded_source = tokenizer.decode(source.tolist(), skip_special_tokens=False)
    decoded_target = tokenizer.decode(target[target != -100].tolist(), skip_special_tokens=False)
    print("Decoded source sequence:", decoded_source)
    print("Decoded target sequence:", decoded_target)

