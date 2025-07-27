import torch.nn as nn
import torch

class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_model, vocab_size, hidden_dim, embedding_dim=1024):
        super().__init__()
        self.embed_model = embed_model
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, dropout=0.2, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embedding_dim, hidden_dim * 2)

        # Attention over encoder outputs
        self.W_denc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.W_denc2 = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)


        # Final vocab projection
        self.U = nn.Linear(hidden_dim * 4, vocab_size)

        self.vocab_size = vocab_size

    def decoder_attention(self, decoder_hidden, encoder_hidden):
        B, T, H = encoder_hidden.shape  # B=batch, T=encoder seq len, H=hidden size
        decoder_exp = decoder_hidden.unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        combined = torch.cat([encoder_hidden, decoder_exp], dim=2)  # (B, T, 2H)
        x = torch.tanh(self.W_denc1(combined))                      # (B, T, H)
        x = torch.tanh(self.W_denc2(x))                             # (B, T, H)
        u = self.w(x).squeeze(-1)                                   # (B, T)
        attn = torch.softmax(u, dim=1)                              # (B, T)

        context = torch.bmm(attn.unsqueeze(1), encoder_hidden).squeeze(1)  # (B, H)
        concat = torch.cat([decoder_hidden, context], dim=1)          # (B, 2H)
        vocab_logits = self.U(concat)                                 # (B, vocab_size)
        return vocab_logits

    def forward(self, inputs, targets, teacher_forcing_ratio=1.0, kb_keys=None, fine_tune=False):
    

        batch_size, T_out, _ = targets.shape

        enc_out, (h_enc, c_enc) = self.encoder(inputs)  # enc_out: (B, T_in, H)
        hidden = torch.cat([h_enc[0], h_enc[-1]], dim=-1)  # (B, hidden_dim*2)
        cell = torch.cat([c_enc[0], c_enc[-1]], dim=-1)    # (B, hidden_dim*2)

        logits = []

        dec_input = targets[:, 0, :]  # Initial decoder input (usually <sos> embedding)

        for t in range(1, T_out):
            hidden_state, cell_state = self.decoder(dec_input, (hidden, cell))
            hidden = hidden_state
            cell = cell_state

            hidden_logits = self.decoder_attention(hidden_state, enc_out)
            logits.append(hidden_logits)

            # Decide whether to do teacher forcing this step
            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher_forcing:
                # Use ground-truth target embedding for next input
                dec_input = targets[:, t, :]
            else:
                # Use model prediction:
                pred_tokens = torch.argmax(hidden_logits, dim=1)  # (B,)
                with torch.no_grad():
                    # Get embeddings for predicted tokens
                    # Assuming embed_model accepts token IDs and returns embeddings
                    dec_input = self.embed_model(pred_tokens)['last_hidden_state'][:, 0, :]

        return torch.stack(logits, dim=1)  # (B, T_out-1, vocab_size)



