import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(self, hidden_dim=768,
                 interm_size=3072, img_size=224,
                 patch_size=16, n_channels=3, n_attention_heads=12,
                 n_hidden_layers=12, layer_norm_eps=1e-6,
                 n_image_tokens=None):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.interm_size = interm_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_attention_heads = n_attention_heads
        self.n_hidden_layers = n_hidden_layers
        self.layer_norm_eps = layer_norm_eps
        self.n_image_tokens = n_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        # image is square
        self.n_patches = (config.img_size // config.patch_size) ** 2

        # [bs, channels, h, w] -> [bs, hidden_dim, n_patches, n_patches]    
        self.patch_embedding = nn.Conv2d(config.n_channels, config.hidden_dim,
                                         config.patch_size, config.patch_size,
                                         padding='valid')

        self.pos_embedding = nn.Embedding(self.n_patches, self.embed_dim)

        self.register_buffer(
            'position_ids',
            torch.arange(self.n_patches).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values):
        patch_embedding = self.patch_embedding(pixel_values)
        # [bs, hidden_dim, n_patches, n_patches] -> [bs, hidden_dim, n_patches ** 2]
        flat_embeddings = torch.flatten(patch_embedding, 2)
        # [bs, hidden_dim, n_patches ** 2] -> [bs, n_patches ** 2, hidden_dim]
        flat_embeddings = flat_embeddings.permute(0, 2, 1)
        # [bs, n_patches ** 2, hidden_dim] + [1, n_patches ** 2, hidden_dim]
        return flat_embeddings + self.pos_embedding(self.position_ids)


class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

        self.use_conn = False
        self.prev = None

    def forward(self, x):
        if self.use_conn:
            return self.prev + x
        else:
            self.prev = x
            self.use_conn = True


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.mlp = nn.Sequential([
            nn.Linear(config.hidden_dim, config.interm_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.interm_size, config.hidden_dim)
        ])

    def forward(self, x):
        return self.mlp(x)


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.n_patches = (config.img_size // config.patch_size) ** 2
        self.n_heads = config.n_attention_heads
        self.head_dim = self.hidden_dim // self.n_heads

        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_o = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, hidden_states):
        bs, seq_len, embed_dim = hidden_states.shape
        
        Q_x = self.W_q(hidden_states)
        K_x = self.W_k(hidden_states)
        V_x = self.W_v(hidden_states)
        # (bs, seq_len, self.n_heads, self.head_dim) -> 
        # (bs, self.n_heads, seq_len, self.head_dim)
        Q_x = Q_x.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_x = K_x.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_x = V_x.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attention_weights = torch.matmul(Q_x, K_x.transpose(2, 3)) * self.scale

        if attention_weights.size() != (bs, self.n_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bs, self.n_heads, seq_len, seq_len)}, but is"
                f"{attention_weights.size()}"
            )

        attention_weights = torch.softmax(attention_weights,
                                          dim=-1, dtype=torch.float32).to(
                                              Q_x.dtype)
        attention_out = torch.matmul(attention_weights, V_x)

        if attention_out.size() != (bs, self.n_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(bs, self.n_heads, seq_len, self.head_dim)}, but is"
                f"{attention_out.size()}"
            )

        attention_out = attention_out.transpose(1, 2).contigious().view(bs, seq_len, self.hidden_dim)
        attention_out = self.W_o(attention_out)
        
        return attention_out, attention_weights


class SiglipVisionEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
       
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim,
                                        eps=config.layer_norm_eps)
        
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim,
                                        eps=config.layer_norm_eps)
        
        self.MLP = SiglipMLP(config)
        self.self_attn = SiglipAttention(config)

    def forward(self, embeddings):
        residual = embeddings 
        x = self.layer_norm1(embeddings)
        x, _ = self.self_attn(x)
        x = x + residual
        
        residual = x

        x = self.layer_norm2(x)
        x = self.MLP(x)
        
        return x + residual


class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.n_layers = config.n_hidden_layers

        self.encoder = nn.ModuleList([
            SiglipVisionEncoderLayer(config) for i in range(self.n_layers)
        ])

    def forward(self, input_embeds):
        hidden_states = input_embeds

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        return hidden_states
    

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_dim

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)

        self.post_layer_norm = nn.LayerNorm(self.embed_dim,
                                            eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        return self.post_layer_norm(
            self.encoder(self.embeddings(pixel_values)))


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        
        super().__init__()

        self.config = config
        self.model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        # [bs, channels, h, w] -> [bs, n_patches, embed_dim]
        return self.model(pixel_values)