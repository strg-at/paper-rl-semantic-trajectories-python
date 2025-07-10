import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)
        

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_trajectory_length=52):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.max_trajectory_length = max_trajectory_length

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        hidden, cell = encoder_hidden  # ⬅️ unpack
        decoder_outputs = []

        if target_tensor is not None:
            decode_steps = target_tensor.size(1)
        else:
            decode_steps = self.max_trajectory_length

        for i in range(decode_steps):
            decoder_output, (hidden, cell) = self.forward_step(decoder_input, (hidden, cell))  # ⬅️ pass tuple
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, (hidden, cell), None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, hidden)
        output = self.out(output)
        return output, (hidden, cell)
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_trajectory_length=52):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_trajectory_length = max_trajectory_length

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        hidden, cell = encoder_hidden  # ⬅️ unpack
        decoder_outputs = []
        attentions = []

        for i in range(self.max_trajectory_length):
            decoder_output, (hidden, cell), attn_weights = self.forward_step(
                decoder_input, (hidden, cell), encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, (hidden, cell), attentions



    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # (batch, 1, hidden_size)

        # Use only hidden state for attention
        query = hidden[0].permute(1, 0, 2)  # (batch, 1, hidden_size)

        context, attn_weights = self.attention(query, encoder_outputs)  # context: (batch, 1, hidden)

        input_lstm = torch.cat((embedded, context), dim=2)  # (batch, 1, 2*hidden)

        output, (hidden, cell) = self.lstm(input_lstm, hidden)
        output = self.out(output)

        return output, (hidden, cell), attn_weights