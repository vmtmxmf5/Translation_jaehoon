import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class Transformer(nn.Module):
    def __init__(self,
                d_model:int = 512,
                nhead:int = 8,
                num_enc_layers:int = 6,
                num_dec_layers:int = 6,
                ff_dim:int = 1024,
                layer_norm_eps:float = 1e-5,
                dropout:float = 0.1,
                src_vocab_size:int = None,
                tgt_vocab_size:int = None):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, ff_dim, dropout)
        # pytorch default eps ==> 분모의 stability를 위해서 더함
        encoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm, d_model, src_vocab_size)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, ff_dim, dropout)
        decoder_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers, decoder_norm, d_model, tgt_vocab_size)

    def forward(self,
                src,
                tgt,
                src_mask = None,
                tgt_mask = None,
#                 memory_mask = None,
#                 src_key_padding_mask = None,
#                 tgt_key_padding_mask = None,
#                 memory_key_padding_mask = None 
                ):
        memory = self.encoder(src, src_mask) #, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
#                             tgt_mask=tgt_mask, # memory mask는 필요없고, memory key padding은 해줘야 src 이상한거 참조 안한다
#                             memory_mask=memory_mask,
#                             tgt_key_padding_mask=tgt_key_padding_mask,
#                             memory_key_padding_mask=memory_key_padding_mask) ### memory key padding 수정
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm, d_model, src_vocab_size):
        super().__init__()
        # 객체를 생성하지 않았으므로 layer를 복사
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.norm = norm
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model) # hid dim == emb dim
        self.src_pos_emb = nn.Embedding(512, d_model) # 우리 모델은 최대 max_length 만큼의 토큰 개수 만큼을 '한 문장'으로 받아들일 수 있다
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, src, src_mask=None, PE=None): #, src_key_padding_mask=None):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        src_pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        src_emb = self.dropout(self.src_tok_emb(src) + self.src_pos_emb(src_pos))    
        
        output = src_emb
        for layer in self.layers:
            output = layer(output, src_mask) #, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim=1024, dropout=0.1, layer_norm_eps=1e-5, device=None):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
    def forward(self, src, src_mask=None): #, src_key_padding_mask=None):
        x = src
        # 레이어 정규화를 먼저하고 합하면 성능이 미미하게 향상
        # 파이토치 튜토리얼에 레퍼런스 있음
        x = x + self._sa_block(self.norm1(x), src_mask) #, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
    def _sa_block(self, x, src_mask=None): #, key_padding_mask=None):
        x = self.self_attn(x, x, x, src_mask)[0]
#                            key_padding_mask=key_padding_mask)[0] # (att. value, att. weight)
        return self.dropout1(x)
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm, d_model, tgt_vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.norm = norm
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model) # hid dim == emb dim
        self.tgt_pos_emb = nn.Embedding(512, d_model) # 우리 모델은 최대 max_length 만큼의 토큰 개수 만큼을 '한 >문장'으로 받아들일 수 있다
        self.dropout = nn.Dropout(0.1)
    def forward(self, tgt, memory, src_mask, tgt_mask):
#                 tgt_mask=None,
#                 memory_mask=None,
#                 tgt_key_padding_mask=None,
#                 memory_key_padding_mask=None):
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(tgt.device)
        tgt_emb = self.dropout(self.tgt_tok_emb(tgt) + self.tgt_pos_emb(tgt_pos))
       
        output = tgt_emb
        for layer in self.layers:
            output = layer(output, memory, src_mask, tgt_mask) 
#                            tgt_mask=tgt_mask,
#                            memory_mask=memory_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask)
        output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim=1024, dropout=0.1, layer_norm_eps=1e-5, device=None):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, src_mask, tgt_mask):
#                 tgt_mask=None,
#                 memory_mask=None,
#                 tgt_key_padding_mask=None,
#                 memory_key_padding_mask=None):
        x = tgt
        x = x + self._sa_block(self.norm1(x), tgt_mask) #, tgt_key_padding_mask)
        x = x + self._ca_block(self.norm2(x), memory, src_mask) #, memory_key_padding_mask)
        x = x + self._ff_block(self.norm3(x))
        return x

    def _sa_block(self, x, mask): #attn_mask=None, key_padding_mask=None):
        x = self.self_attn(x, x, x, mask)[0]
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)
    def _ca_block(self, x, mem, mask): # attn_mask=None, key_padding_mask=None):
        x = self.multihead_attn(x, mem, mem, mask)[0]
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            memory=True)[0]
        return self.dropout2(x)
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    def forward(self, query, key, value, mask,): #attn_mask=None, key_padding_mask=None, memory=False):
        # query : [batch size, src time steps, hid dim]
        # key : [batch size, src time steps, hid dim]
        # value : [batch size, src time steps, hid dim]
        batch_size = query.shape[0]
        Q = self.fc_q(query) # [batch size, src time steps, hid dim]
        K = self.fc_k(key) 
        V = self.fc_v(value) 
        
        # [batch size, n_heads, src time stpes, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device) # Attention Score
        # [batch size, n_heads, src time steps, src time steps]
        
        # padding만 False, 원본은 True
#         if key_padding_mask is not None:
#             # mini batch에 padding이 있다
#             key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
#             # (B, 1, 1, T)
#             if attn_mask is not None:
#                 # 미래 정보 사전관찰 방지 attn_mask
#                 # (B, 1, T, T)
#                 if memory == True:
#                     subsequent_mask = attn_mask & key_padding_mask.type(torch.bool)
#                     subsequent_mask = subsequent_mask.float() # (B, 1, Tdec, Tenc)
#                 else:
#                     tgt_mask = attn_mask & key_padding_mask.type(torch.bool)
#                     subsequent_mask = tgt_mask.float()
#                 energy = energy.masked_fill(subsequent_mask == 0, float('-inf'))
#             else:
#                 key_padding_mask = key_padding_mask.float()
#                 energy = energy.masked_fill(key_padding_mask == 0, float('-inf'))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
            # print(energy)
        att_weights = torch.softmax(energy, dim=-1) # [batch size, n_heads, src time steps, src time steps]
        att_values = torch.matmul(self.dropout(att_weights), V)
        att_values = att_values.permute(0, 2, 1, 3).contiguous()
        # att_values : [batch size, src time steps, n heads, head dim]
        att_values = att_values.view(batch_size, -1, self.d_model) # view로 concat을 대체한다
        # att_values : [batch size, src time steps, hid dim]
        att_values = self.fc_o(att_values)
        return att_values, att_weights


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size:int, dropout:float, max_len:int = 5000):
        super().__init__()
        # Transformer PE : cos(pos / 10000^(2i/d_model)) or sin(pos / 10000^(2i/d_model))
        # EXP[-2i*log(10000)/d_model] == 10000^(-2i/d_model) == 1 / 10000^(2i/d_model)
        den = torch.exp(-torch.arange(0, emb_size, 2)*math.log(10000)/emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1) # 브로드캐스팅을 위해서 차원 추가
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # 짝수 임베딩 디멘젼에 sin 정보 
        pos_embedding[:, 1::2] = torch.cos(pos * den) # 홀수 임베딩 디멘젼에 cos 정보
        self.pos_embedding = pos_embedding.unsqueeze(0) # 배치 차원 추가 (B, T, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding): # (B, T, d_model)
    # 배치만큼 브로드 캐스팅이 일어나서 위치 정보를 더해줄 것
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :].to(token_embedding.device))


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2seqTransformer(nn.Module):
    def __init__(self,
                src_vocab_size: int,
                tgt_vocab_size: int,
                num_enc_layers: int = 6,
                num_dec_layers: int = 6,
                emb_size: int = 512,
                nhead: int = 8,
                ff_dim: int = 512,
                dropout: float = 0.1,
                ):
        super().__init__()
        self.transformer = Transformer(d_model=emb_size,
                                    nhead=nhead,
                                    num_enc_layers=num_enc_layers,
                                    num_dec_layers=num_dec_layers,
                                    ff_dim=ff_dim,
                                    dropout=dropout,
                                    src_vocab_size=src_vocab_size,
                                    tgt_vocab_size=tgt_vocab_size)
        self.generator = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, tgt_vocab_size),
            nn.LogSoftmax(dim=-1)
        )
            
        self.scale = torch.sqrt(torch.FloatTensor([emb_size]))
        self.dropout = nn.Dropout(dropout)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        # self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size) # hid dim == emb dim
        # self.src_pos_emb = nn.Embedding(512, emb_size) # 우리 모델은 최대 max_length 만큼의 토큰 개수 만큼을 '한 문장'으로 받아들일 수 있다
        # self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size) # hid dim == emb dim
        # self.tgt_pos_emb = nn.Embedding(512, emb_size) # 우리 모델은 최대 max_length 만큼의 토큰 개수 만큼을 '한 >문장'으로 받아들일 수 있다
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask): # , src_padding_mask, tgt_padding_mask, memory_mask):
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        # tgt_len = tgt.shape[1]                  
        # src_pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        # tgt_pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        # src_emb = self.dropout((self.src_tok_emb(src) * self.scale.to(src.device)) + self.src_pos_emb(src_pos))
        # tgt_emb = self.dropout((self.tgt_tok_emb(tgt) * self.scale.to(src.device)) + self.tgt_pos_emb(tgt_pos))
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        logits = self.transformer(src = src,
                                  tgt = tgt,
                                  src_mask = src_mask,
                                  tgt_mask = tgt_mask)
#                                   memory_mask = memory_mask,
#                                   src_key_padding_mask = src_padding_mask,
#                                   tgt_key_padding_mask = tgt_padding_mask,
#                                   memory_key_padding_mask = src_padding_mask)
        return self.generator(logits) ## rev.
    
    def search(self, src, src_mask, max_length=20, bos_id=2, eos_id=3):
        
        BOS_token = bos_id
        EOS_token = eos_id

        y_hats, indice = [], []
        with torch.no_grad():
            # ENCODER : src = (T)
            # src_emb = self.positional_encoding(self.src_tok_emb(src.reshape(1, -1))) # 배치 추가
            # batch_size = 1
            # src_len = src.shape[0]
            
            # src_pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
            # src_emb = self.src_tok_emb(src.reshape(1, -1)) * self.scale + self.src_pos_emb(src_pos)
            
            ######### batch size 1 #######################
            memory = self.transformer.encoder(src.reshape(1, -1),
                                              src_mask=src_mask)
                                              # src_key_padding_mask=None)
            # DECODER : BOS 토큰부터 넣기 시작
            dec_input = torch.LongTensor([[BOS_token]])
            dec_input_len = torch.LongTensor([dec_input.size(-1)])
            
            for t in range(max_length):
                # tgt_pos = torch.arange(0, dec_input_len.item()).unsqueeze(0).repeat(batch_size, 1)
                # tgt = self.tgt_tok_emb(dec_input) * self.scale + self.tgt_pos_emb(torch.LongTensor(tgt_pos))
                # mask = (torch.triu(torch.ones(tgt.size(1), tgt.size(1))) == 1).transpose(0, 1)
                # tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                
                ## rev.
                logits = self.transformer.decoder(dec_input,
                                                  memory,
                                                  src_mask=src_mask,
                                                  tgt_mask=None)
                                                #  tgt_key_padding_mask=None,
                                                #  memory_key_padding_mask=None)
                output = self.generator(logits) ## rev.
                
                next_item = output.topk(5)[1].view(-1)[-1].item()
                next_item = torch.tensor([[next_item]])

                dec_input = torch.cat([dec_input, next_item], dim=-1)
                # print("({}) dec_input: {}".format(di, dec_input))

                dec_input_len = torch.LongTensor([dec_input.size(-1)])
                
                if next_item.view(-1).item() == EOS_token:
                    break
        
        return dec_input.view(-1).tolist()[1:]


def create_mask(src, tgt, pad_id, device):
    # 배치 고려
    src_time_steps = src.shape[1]
    tgt_time_steps = tgt.shape[1]     
    tgt_sub_mask = torch.tril(torch.ones((tgt_time_steps, tgt_time_steps), device=device)).bool()
    # src_mask = torch.zeros((src_time_steps, src_time_steps), device=device).type(torch.bool)

    # QK^T 의 shape을 맞춰야 하므로 row는 decoder time steps, col은 enc time steps를 넣으면 된다
    # memory_mask = torch.triu(torch.ones((src_time_steps, tgt_time_steps), device=device)).bool()
    src_padding_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_padding_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_sub_mask & tgt_padding_mask
    return src_padding_mask, tgt_mask #, tgt_padding_mask, memory_mask 

if __name__=='__main__':
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    
    model = Seq2seqTransformer(10, 10)
    
    # Transformer Info.
    # for layer in model.state_dict():
    #     print(layer, '\t', model.state_dict()[layer].size())
    
    # Prediction
    src_tmp = torch.LongTensor([[4, 6, 7, 8, 3, 1],
                                [7, 6, 5, 9, 4, 3]])
    tgt_tmp = torch.LongTensor([[5, 5, 7, 8, 3, 1],
                                [7, 4, 5, 4, 4, 3]])

    src_mask, tgt_mask = create_mask(src_tmp, tgt_tmp, 1, None)
    pred = model(src_tmp, tgt_tmp, src_mask, tgt_mask)#, src_padding_mask, tgt_padding_mask, memory_mask)
    # print(pred)
