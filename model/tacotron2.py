import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 256]):
        super().__init__()
        layers = []
        for size in sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_dim = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Postnet(nn.Module):
    def __init__(self, mel_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(mel_dim, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, mel_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(mel_dim),
        )

    def forward(self, x):
        return self.convs(x.transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        packed = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.query_proj = nn.Linear(decoder_dim, decoder_dim)
        self.key_proj = nn.Linear(encoder_dim, decoder_dim)
        self.energy_proj = nn.Linear(decoder_dim, 1)

    def forward(self, query, keys, values):
        # query: [B, decoder_dim], keys: [B, T, encoder_dim]
        query = self.query_proj(query).unsqueeze(1)  # [B, 1, D]
        keys = self.key_proj(keys)  # [B, T, D]
        energy = torch.tanh(query + keys)  # [B, T, D]
        attention = self.energy_proj(energy).squeeze(-1)  # [B, T]
        weights = torch.softmax(attention, dim=-1)  # [B, T]
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # [B, encoder_dim]
        return context, weights


class Decoder(nn.Module):
    def __init__(self, mel_dim=80, encoder_dim=512, prenet_sizes=[256, 256]):
        super().__init__()
        self.prenet = Prenet(mel_dim, prenet_sizes)
        self.attention = Attention(encoder_dim, 1024)
        self.rnn1 = nn.LSTMCell(prenet_sizes[-1] + encoder_dim, 1024)
        self.rnn2 = nn.LSTMCell(1024, 1024)
        self.linear = nn.Linear(1024 + encoder_dim, mel_dim)

    def forward(self, encoder_outputs, decoder_inputs):
        B, T, _ = decoder_inputs.size()
        h1, c1 = torch.zeros(B, 1024, device=decoder_inputs.device), torch.zeros(B, 1024, device=decoder_inputs.device)
        h2, c2 = torch.zeros(B, 1024, device=decoder_inputs.device), torch.zeros(B, 1024, device=decoder_inputs.device)
        context = torch.zeros(B, encoder_outputs.size(2), device=decoder_inputs.device)
        outputs = []

        for t in range(T):
            prenet_out = self.prenet(decoder_inputs[:, t, :])
            att_input = torch.cat((prenet_out, context), dim=-1)
            h1, c1 = self.rnn1(att_input, (h1, c1))
            h2, c2 = self.rnn2(h1, (h2, c2))
            context, _ = self.attention(h2, encoder_outputs, encoder_outputs)
            out = self.linear(torch.cat((h2, context), dim=-1))
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class Tacotron2(nn.Module):
    def __init__(self, vocab_size, mel_dim=80):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(mel_dim)
        self.postnet = Postnet(mel_dim)

    def forward(self, text_inputs, mel_inputs, text_lengths):
        encoder_outputs = self.encoder(text_inputs, text_lengths)
        mel_out = self.decoder(encoder_outputs, mel_inputs)
        mel_post = mel_out + self.postnet(mel_out)
        return mel_out, mel_post, None, None
