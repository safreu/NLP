from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from configuration.config import SEED

SentencePair = tuple[str, str]


class TransformerDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[SentencePair],
        tokenizer: PreTrainedTokenizerBase,
        max_length=256,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]

        return (
            torch.tensor(self.tokenize(source), dtype=torch.long),
            torch.tensor(self.tokenize(target), dtype=torch.long),
        )


def make_collate_fn(tokenizer: PreTrainedTokenizerBase):
    def collate_fn(batch):
        source_batch, target_batch = zip(*batch, strict=True)

        source_batch = pad_sequence(
            source_batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        target_batch = pad_sequence(
            target_batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        source_attention_mask = (source_batch != tokenizer.pad_token_id).long()

        return source_batch, source_attention_mask, target_batch

    return collate_fn


def create_data_loader(
    pairs: Sequence[SentencePair],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TransformerDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    generator = None

    if shuffle:
        generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer),
        generator=generator,
    )


def train_model(model, train_loader, optimizer, criterion, device, num_epochs) -> None:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for source_batch, source_attention_mask, target_batch in train_loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            source_attention_mask = source_attention_mask.to(device)

            decoder_input = target_batch[:, :-1]
            targets = target_batch[:, 1:]

            output = model(source_batch, source_attention_mask, decoder_input)

            output = output.reshape(-1, output.shape[-1])
            targets = targets.reshape(-1)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
