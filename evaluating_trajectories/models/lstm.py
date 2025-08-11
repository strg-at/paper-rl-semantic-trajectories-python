import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    """
    Encoder takes continuous inputs (B, T, D) and returns:
      - encoder_outputs: (B, T, H)
      - h: (B, H) latent for δ/β̂
    """
    def __init__(self, input_dim: int, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)

    def forward(self, x):                 # x: (B, T, D)
        x = self.dropout(x)
        outputs, (h, _) = self.lstm(x)    # outputs: (B,T,H), h: (1,B,H)
        return outputs, h[-1]             # (B,T,H), (B,H)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys):  # query: (B,1,H), keys: (B,T,H)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (B,T,1)
        weights = F.softmax(scores.squeeze(-1), dim=-1).unsqueeze(1)  # (B,1,T)
        context = torch.bmm(weights, keys)                             # (B,1,H)
        return context, weights                                        # (B,1,H), (B,1,T)


class AttnDecoderRNN(nn.Module):
    """
    Attention decoder that reconstructs continuous outputs (B, T, D).
    - At each step, query = current decoder hidden (B,1,H)
    - Attend over encoder_outputs (B,T,H)
    - LSTM input = concat(repeated latent h, context)  -> (B,1,2H)
    - Output projected to D, optimized with masked MSE.
    """
    def __init__(self, hidden_size: int, output_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.attn = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, latent_h, out_len: int):
        B, T, H = encoder_outputs.size()
        # init hidden/cell to zeros (or learnable); using zeros is common
        h0 = torch.zeros(1, B, H, device=encoder_outputs.device)
        c0 = torch.zeros(1, B, H, device=encoder_outputs.device)
        hidden = (h0, c0)

        # we’ll feed the SAME latent_h at every step (like the paper), but also add attention context
        repeated_h = latent_h.unsqueeze(1)  # (B,1,H)

        outputs = []
        for _ in range(out_len):
            query = hidden[0].permute(1, 0, 2)            # (B,1,H)
            context, _ = self.attn(query, encoder_outputs) # (B,1,H)

            lstm_in = torch.cat([repeated_h, context], dim=-1)  # (B,1,2H)
            lstm_in = self.dropout(lstm_in)

            y, hidden = self.lstm(lstm_in, hidden)        # y: (B,1,H)
            y = self.proj(y)                              # (B,1,D)
            outputs.append(y)

        return torch.cat(outputs, dim=1)
    