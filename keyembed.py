
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_key_embed(keys, tokenizer, embed_model):
    keys_embed = []
    for key in keys:
        key = tokenizer(key, return_tensors='pt')['input_ids'][:,:-1]
        key_embed = embed_model(key.to(device), )['last_hidden_state'][0]
        keys_embed.append(key_embed)

    keys_embed = torch.stack(keys_embed, dim=0).squeeze(1).detach()

    return keys_embed

