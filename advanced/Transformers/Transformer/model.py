import torch
import torch.nn as nn
import math 


class EmbeddingBlock(nn.Module):
    def __init__(self, vocab_sz, d_model):
        super(EmbeddingBlock, self).__init__()

        self.d_model = d_model
        self.vocab_sz = vocab_sz
        self.embedding = nn.Embedding(vocab_sz, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, seq_sz, dropout_rate):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model
        self.seq_sz = seq_sz

        # Matrix for pos. encoding of (seq_sz, d_model)
        PE = torch.zeros(seq_sz, d_model)
        # positions vector of shape (seq_sz, 1)
        pos = torch.arange(0, seq_sz, dtype=torch.float).unsqueeze(1)
        # Same denominator expression for both sin and cos
        denominator = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(pos * denominator)
        PE[:, 1::2] = torch.cos(pos * denominator)

        # Add batch dimension --> (1, seq_sz, d_model)
        PE = PE.unsqueeze(0)

        # Register PE as a buffer
        self.register_buffer('PE', PE)

    def forward(self, x):
        return x + self.PE[:, :x.shape[1], :].requires_grad_(False)
    

class LayerNormBlock(nn.Module):
    def __init__(self, eps=1e-8):
        super(LayerNormBlock, self).__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape - (bs, seq_sz, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForwardLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.layer(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert d_model % h == 0, "d_model isn't divisible by h"
        self.d_v = d_model // h
        self.d_model = d_model        
        self.h = h

        self.dropout = nn.Dropout(dropout_rate)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def self_attention(Q, K, V, dropout_rate=None, mask=None):
        d_k = Q.shape[-1]

        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout_rate is not None:
            attention_scores = nn.Dropout(dropout_rate)(attention_scores)
        return (attention_scores @ V), attention_scores
    
    def forward(self, Q, K, V, mask=None, dropout_rate=None):
        Q_w = self.W_q(Q)  # shape - (batch, seq_sz, d_model) 
        K_w = self.W_k(K)
        V_w = self.W_v(V)
        
        # for each of them:
        # (batch, seq_sz, d_model) -> (batch, seq_sz, h, d_v)
        # -> (batch, h, seq_sz, d_v)
        Q_w = Q_w.view(Q_w.shape[0], Q_w.shape[1],
                       self.h, self.d_v).transpose(1, 2)
        K_w = K_w.view(K_w.shape[0], K_w.shape[1],
                       self.h, self.d_v).transpose(1, 2)
        V_w = V_w.view(V_w.shape[0], V_w.shape[1],
                       self.h, self.d_v).transpose(1, 2)
        
        x, self.attention_scores = self.self_attention(
            Q_w, K_w, V_w, dropout_rate, mask)

        # (batch, h, seq_sz, d_v) -> (batch, seq_sz, h, d_v) -> 
        # (batch, seq_sz, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.W_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNormBlock()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, attention_block, ff_layer, dropout_rate):
        super(EncoderBlock, self).__init__()

        self.attention_block = attention_block
        self.ff_layer = ff_layer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout_rate) for _ in range(2)])
        self.dropout_rate = dropout_rate

    # WE need masks to mask padding (padding tokens)    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.attention_block(x, x, x,
                                              src_mask, self.dropout_rate))
        x = self.residual_connections[1](x, self.ff_layer)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormBlock()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block,
                 ff_layer, dropout_rate):
        super(DecoderBlock, self).__init__()

        self.attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_layer = ff_layer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout_rate) for _ in range(3)])
        self.dropout_rate = dropout_rate

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.attention_block(x, x, x,
                                              tgt_mask, self.dropout_rate))
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(
                x, encoder_out, encoder_out, src_mask, self.dropout_rate))
        x = self.residual_connections[2](x, self.ff_layer)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormBlock()

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)


# name it projection cause it projects d_model -> vocab. size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, dict_size):
        super(ProjectionLayer, self).__init__()
        self.layer = nn.Linear(d_model, dict_size)

    def forward(self, x):
        # log_softmax - gives better num. stability
        return torch.log_softmax(self.layer(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: EncoderBlock, decoder: DecoderBlock,
                 src_embed: EmbeddingBlock, tgt_embed: EmbeddingBlock,
                 src_pos: PositionalEncoder, tgt_pos: PositionalEncoder,
                 projection_layer: ProjectionLayer):
        
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocal_sz, tgt_vocab_sz, src_seq_sz, tgt_seq_sz,
                      d_model=512, N=6, h=8, dropout_rate=0.1, d_ff=2048):
    # Embedding blocks
    src_embed = EmbeddingBlock(src_vocal_sz, d_model)
    tgt_embed = EmbeddingBlock(tgt_vocab_sz, d_model)

    # Pos. encoding blocks
    src_pos = PositionalEncoder(d_model, src_seq_sz, dropout_rate)
    tgt_pos = PositionalEncoder(d_model, tgt_seq_sz, dropout_rate)

    # Encoder blocks repeated N times
    encoder_blocks = []
    for _ in range(N):
        enc_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout_rate)
        
        enc_ff_layer = FeedForwardLayer(d_model, d_ff, dropout_rate)
        encoder = EncoderBlock(enc_self_attention_block,
                               enc_ff_layer, dropout_rate)
        encoder_blocks.append(encoder)

    # Decoder blocks repeated N times
    decoder_blocks = []
    for _ in range(N):
        dec_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout_rate)
        dec_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout_rate)
        
        dec_ff_layer = FeedForwardLayer(d_model, d_ff, dropout_rate)

        decoder = DecoderBlock(
            dec_self_attention_block, dec_cross_attention_block,
            dec_ff_layer, dropout_rate)
        
        decoder_blocks.append(decoder)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_sz)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos,
                              tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def test_transformer():
    # Define parameters
    src_vocab_sz = 100
    tgt_vocab_sz = 100
    src_seq_sz = 10
    tgt_seq_sz = 10
    d_model = 512
    N = 2
    h = 8
    dropout_rate = 0.1
    d_ff = 2048

    # Build the transformer
    transformer = build_transformer(src_vocab_sz, tgt_vocab_sz, src_seq_sz, tgt_seq_sz,
                                    d_model, N, h, dropout_rate, d_ff)
    
    # Create random input tensors
    src = torch.randint(0, src_vocab_sz, (2, src_seq_sz))
    tgt = torch.randint(0, tgt_vocab_sz, (2, tgt_seq_sz))
    
    # Dummy masks (no masking applied)
    src_mask = torch.ones(2, src_seq_sz).unsqueeze(1).unsqueeze(2)
    tgt_mask = torch.ones(2, tgt_seq_sz).unsqueeze(1).unsqueeze(2)
    
    # Forward pass through the transformer
    encoder_out = transformer.encode(src, src_mask)
    decoder_out = transformer.decode(encoder_out, src_mask, tgt, tgt_mask)
    output = transformer.project(decoder_out)

    print("Test passed: Transformer forward pass successful")
    print(f"Output shape: {output.shape}")

if __name__ == '__main__':
    # Run the test
    test_transformer()