from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import optim
from torch.utils.data import DataLoader, Dataset

from configuration.config import SEED
from data.wikilarge_loader import WikiLargeLoader
from evaluation.metrics_builder import compute_all_metrics
from storage.json_store import write_json


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] #number of examples in the batch
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

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N,
            query_len,
            self.heads * self.head_dim
        )  # after einsum (N,query_len, head, head_dim) then flatten  the last two dimensions (N, query_len, embed_size)

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
            nn.Linear(forward_expansion * embed_size, embed_size)
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
        max_length
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
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
    def __init__(self, embed_size, heads, dropout, forward_expansion,device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
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
        max_length
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion,device)
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
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
    
class TransformerDataset(Dataset):
    def __init__(self, pairs, vocabulary):
        self.pairs = pairs
        self.vocab = vocabulary
        
    def tokenize(self, text):
        return [self.vocab["<sos>"]] +[ 
                    self.vocab.get(token, self.vocab["<unk>"]) 
                    for token in text.split()
                ] + [self.vocab["<eos>"]]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        source, target = self.pairs[index]
        return torch.tensor(self.tokenize(source)), torch.tensor(self.tokenize(target))
    

vocabulary = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}


def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch = pad_sequence(
        source_batch,
        batch_first=True,
        padding_value=vocabulary["<pad>"]
    )
    
    target_batch = pad_sequence(
        target_batch,
        batch_first=True,
        padding_value=vocabulary["<pad>"]
    )
    return source_batch, target_batch


def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for source_batch, target_batch in train_loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            decoder_input = target_batch[:, :-1]
            targets =  target_batch[:, 1:]
            
            output = model(source_batch, decoder_input)
            
            output = output.reshape(-1, output.shape[-1])
            targets = targets.reshape(-1)
            
            loss = criterion(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def generate_prediction(model, src_tensor, vocabulary, inv_vocab, device, max_length=30):
    model.eval()

    outputs = [vocabulary["<sos>"]]

    with torch.no_grad():
        for _ in range(max_length):
            trg_input = torch.tensor([outputs]).to(device)
            
            out = model(src_tensor, trg_input)
            
            best_next_item = out.argmax(dim=2)[:, -1].item()
            
            outputs.append(best_next_item)
            
            if best_next_item == vocabulary["<eos>"]:
                break

    return [inv_vocab[idx] for idx in outputs]


def eval_model(model, data_loader, vocabulary, inv_vocab, device, max_length, output_path):
    model.eval()
    
    sources = []
    candidates = []
    references = []
    
    dataset = data_loader.dataset
    
    
    for source, reference in dataset.pairs:
        src_indices = dataset.tokenize(source)
        src_tensor = torch.tensor([src_indices]).to(device)

        prediction_tokens = generate_prediction(
            model=model,
            src_tensor=src_tensor,
            vocabulary=vocabulary,
            inv_vocab=inv_vocab,
            device=device,
            max_length=max_length,
        )
        
        prediction = " ".join(
            token for token in prediction_tokens
            if token not in {"<sos>", "<eos>", "<pad>"}
        )
        
        sources.append(source)
        candidates.append(prediction)
        references.append(reference)
        
    scores = compute_all_metrics(
        sources=sources,
        candidates=candidates,
        references=references,
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_json(scores, output_path)
    
    return scores 
        
        
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, valid, test = WikiLargeLoader(
        max_train_samples=10000, 
        max_eval_samples=2000
    ).load_pairs(False)
    
    for src, target in train:
        for token in src.split() + target.split():
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    
    inv_vocab = {v: k for k, v in vocabulary.items()}

    train_dataset = TransformerDataset(train, vocabulary)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = Transformer(
        src_vocab_size=len(vocabulary), 
        trg_vocab_size=len(vocabulary), 
        src_pad_idx=vocabulary["<pad>"], 
        trg_pad_idx=vocabulary["<pad>"],
        device=device,
        max_length=256,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=10
    ) 
   
    test_dataset = TransformerDataset(test, vocabulary)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
   
    scores = eval_model(
        model=model,
        data_loader=test_loader,
        vocabulary=vocabulary,
        inv_vocab=inv_vocab,
        device=device,
        max_length=30,
        output_path="runs/custom_transformer/wikilarge/base/scores.json",
    ) 
    
    print("Evaluation finished")
    print(scores["sari"]) 