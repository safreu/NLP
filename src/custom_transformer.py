import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from configuration.config import SEED
from data.dataset_loader import DatasetLoader
from data.newsela_loader import NewselaLoader
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from evaluation.asset_sari_evaluator import AssetSariEvaluator
from evaluation.metrics_builder import compute_all_metrics
from storage.json_store import write_json
from storage.paths import RunPaths


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # number of examples in the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # after einsum (N,query_len, head, head_dim)
        # then flatten  the last two dimensions (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        trg_pad_idx,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=256,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        embed_size = self.encoder.config.hidden_size

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src_ids, src_attention_mask, trg):
        src_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)
        
        enc_src = self.encoder(
            input_ids=src_ids,
            attention_mask=src_attention_mask,
        ).last_hidden_state
        
        trg_mask = self.make_trg_mask(trg)
        return self.decoder(trg, enc_src, src_mask, trg_mask)


class TransformerDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )["input_ids"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        return torch.tensor(self.tokenize(source)), torch.tensor(self.tokenize(target))


def make_collate_fn(tokenizer):
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


def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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


def generate_prediction(model, src_tensor, tokenizer, device, max_length=256):
    model.eval()

    outputs = [tokenizer.cls_token_id]
    
    with torch.no_grad():
        for _ in range(max_length):
            trg_input = torch.tensor([outputs]).to(device)

            src_attention_mask = (src_tensor != tokenizer.pad_token_id).long()
            
            out = model(src_tensor, src_attention_mask, trg_input)

            best_next_item = out.argmax(dim=2)[:, -1].item()

            outputs.append(best_next_item)

            if best_next_item == tokenizer.sep_token_id:
                break

    return tokenizer.decode(outputs, skip_special_tokens=True)


def build_custom_transformer_predict_fn(model, tokenizer, device, max_length: int=256):
    def predict_fn(sources: list[str]) -> list[str]:
        predictions = []

        for source in sources:
            src_indices = tokenizer(
                source,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )["input_ids"]

            src_tensor = torch.tensor([src_indices]).to(device)

            prediction = generate_prediction(
                model=model,
                src_tensor=src_tensor,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
            )

            predictions.append(prediction)

        return predictions

    return predict_fn


def eval_model(model, data_loader, tokenizer, device, max_length):
    model.eval()

    sources = []
    candidates = []
    references = []

    dataset = data_loader.dataset

    for source, reference in dataset.pairs:
        src_indices = dataset.tokenize(source)
        src_tensor = torch.tensor([src_indices]).to(device)

        prediction = generate_prediction(
            model=model,
            src_tensor=src_tensor,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

        sources.append(source)
        candidates.append(prediction)
        references.append(reference)

    return compute_all_metrics(
        sources=sources,
        candidates=candidates,
        references=references,
    )


if __name__ == "__main__":
    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=100, max_eval_samples=20),
        #WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        #OneStopLoader(),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    asset_evaluator = AssetSariEvaluator(
        split="validation",
        max_examples=0,
    )

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    for dataset in dataset_loaders:
        start = time.time()

        train, valid, test = dataset.load_pairs(False)

        train_dataset = TransformerDataset(train, tokenizer, 256)

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=make_collate_fn(tokenizer),
            generator=torch.Generator().manual_seed(SEED),
        )

        model = Transformer(
            trg_vocab_size=tokenizer.vocab_size,
            trg_pad_idx=tokenizer.pad_token_id,
            device=device,
            max_length=256,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=10,
        )

        test_dataset = TransformerDataset(test, tokenizer, 256)

        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, collate_fn=make_collate_fn(tokenizer)
        )

        scores = eval_model(
            model=model,
            data_loader=test_loader,
            tokenizer=tokenizer, 
            device=device,
            max_length=256,
        )

        run_paths = RunPaths.for_runs_root(Path(f"runs/custom_transformer/{dataset.name}_1.0"))
        run_paths.pipeline_dir = Path("base")

        predict_fn = build_custom_transformer_predict_fn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=256,
        )

        asset_results = asset_evaluator.run(predict_fn=predict_fn, output_dir=run_paths.output_dir)

        scores.update(asset_results)

        write_json(scores, run_paths.scores_path)

        print(f"Evaluation  of {dataset.name} finished")
        print(scores["sari"])
        print(scores["asset_sari"])
        print(f"{dataset.name} finished in {(time.time() - start) / 60:.1f} minutes")

    print("Complete Evaluation finished")
