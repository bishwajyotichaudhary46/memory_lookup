import argparse
import json
from modelwithkbattention import EncoderDecoderWithKBAttention
import torch
from transformers import AutoTokenizer, AutoModel
from preproces_data import preprocess_data
from load_kb_data import getkeyvaluedata
from tqdm import tqdm

# check cuda is available or not
print(f"check cuda avalable or not : {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("*"*20)


# load kb data
keys, preprocess_kv_data = getkeyvaluedata('appointments_data.json')
print("KB data loaded")

print("*"*20)

print("Downloading pretrianed model")
# download pretrained qwen model 
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
embed_model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')

# add new tokens in qwen model
tokenizer.add_tokens(keys)
# resize embed model dimension after new token added
embed_model.resize_token_embeddings(len(tokenizer))
print("Dowloaded!!")



print("*"*20)

# load key2tokenizer
with open('artifacts/key2tokenizerid.json', 'r') as f:
    key2tokenid = json.load(f)

print("key keb",len(key2tokenid))
# get vocab size
vocab_size = len(tokenizer.vocab)

print("Model Intializing ")
#model initializer
model = EncoderDecoderWithKBAttention(embed_model=embed_model, vocab_size=vocab_size, hidden_dim=320,new2old=key2tokenid, embedding_dim=1024).to(device)
# load weight of trained model'
state_dict = torch.load("models/model_weight_kb_epoch_19.pth")
load_result = model.load_state_dict(state_dict)
print("*"*20)


print("Loading Keys Embeddings")
#  load keys_embed from artficats
keys_embed = torch.load('artifacts/keyembedding.pt')
print("*"*20)



def inference(model=model, tokenizer=tokenizer, embed_model=embed_model,preprocess_data=preprocess_data, keys_embed=keys_embed, device=device, ):
    query = input("Enter what is you query? \n")
    # get start tokens
    start_token = tokenizer(
                '<|im_start|>',
                max_length= 8192,
                return_tensors="pt",
            )['input_ids'][:,0].unsqueeze(0).to(device)
    
    embed_model = embed_model.to(device)

    start_embed = embed_model(start_token)['last_hidden_state'][:, 0, :].to(device)

    input_tokens = tokenizer(
        query,
        max_length= 8192,
        return_tensors="pt",
        )['input_ids'][:, :-1].to(device)
    
    embed_query = embed_model(input_tokens)['last_hidden_state']
    enc_out, (h_enc, c_enc) = model.encoder(embed_query)
    hidden = torch.cat([h_enc[0], h_enc[1]], dim=-1)  
    cell   = torch.cat([c_enc[0], c_enc[1]], dim=-1) 
    tokens = []
    stop_token = -1
    input_emb = start_embed
   
    print("Inference Start")
    for i in tqdm(range(20), desc="Inferencing"):
        
        hidden, cell = model.decoder(input_emb, (hidden, cell))

        vocab_logits = model.decoder_attention(hidden, enc_out)  

        kb_attention_logits = model.kb_attention(hidden, keys_embed).to('cuda')
                
        hidden_logits = kb_attention_logits + vocab_logits
        pred_tokens = torch.argmax(hidden_logits, dim=1) 
        stop_token = pred_tokens.item()
        if pred_tokens.item() == 151643:
            break
        tokens.append(pred_tokens)

        input_emb = embed_model(pred_tokens.unsqueeze(0))['last_hidden_state'][0]


    pred_tokens = tokenizer.decode(torch.stack(tokens,  dim=1)[0])

    text = ""
    for id , token in enumerate(pred_tokens.split()):
        if token in keys:
            text += preprocess_kv_data[token]
        else:
            if id != 0:
                if pred_tokens.split()[id -1 ] != token:
                    text += token
            else:       
                text += token
        text += " "

    

    return text, query



    

if __name__ == "__main__":
    result, query = inference()

    print("*"*20)
    print(f"\nQuestion : {query}")

    print(f"Answer : {result} \n")
