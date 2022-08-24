from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from .autoencoder_xl import *


class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)


    def forward(self, commands, args, groups=None):
        S, N = commands.shape
        
        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL
        
        if self.use_group:
            src = src + self.group_embed(groups.long())
        
        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)
        
    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        self.n_token = seq_len
        self.div_n = cfg.enc_div_val
        self.cutoffs = [self.n_token // 2]
        self.tie_projs = [False] + [True] * len(self.cutoffs)
        self.encoder = MemTransformerLM(n_token=self.n_token, n_layer=cfg.n_layers, n_head=cfg.n_heads,
                            d_model=cfg.d_model, d_head=cfg.d_head, d_inner=cfg.dim_feedforward, dropout=cfg.dropout,
                            dropatt=cfg.dropatt, tie_weight=True, d_embed=cfg.d_embed, div_val=cfg.enc_div_val,
                            tie_projs=self.tie_projs, pre_lnorm=True, tgt_len=cfg.enc_tgt_len, ext_len=cfg.enc_ext_len,
                            mem_len=cfg.enc_mem_len, cutoffs=self.cutoffs, attn_type=cfg.attn_type)          
        
    def forward(self, commands, args, *mems):
        # commands.shape: [Nc, batch_size]
        # args.shape: [Nc, batch_size, N_args]
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        # src.shape: [Nc, batch_size, d_model]
        src = self.embedding(commands, args, group_mask)
        memory = []
        for d in range(self.div_n):
            start = int(self.n_token / self.div_n) * d
            end = int(self.n_token / self.div_n) * (d + 1)
            memory_d, mems = self.encoder(src[start:end], src[start:end], *mems, d=d)            
            memory.append(memory_d)
            
        memory = torch.cat(memory, dim=0)
        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        
        if mems is not None:
            return z, mems
        else:
            return z

class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        seq_len = cfg.max_total_len
        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)
        #self.embedding = nn.Linear(cfg.d_model, cfg.d_model * seq_len)
        
        # 22_04_11 수정
        # transformer -> transformer-xl 변경
        self.n_token = seq_len
        self.d_model = cfg.d_model
        self.div_n = cfg.dec_div_val
        self.cutoffs = [self.n_token // 2]
        self.tie_projs = [False] + [True] * len(self.cutoffs)
        self.decoder = MemTransformerLM(n_token=self.n_token, n_layer=cfg.n_layers, n_head=cfg.n_heads,
                            d_model=cfg.d_model, d_head=cfg.d_head, d_inner=cfg.dim_feedforward, dropout=cfg.dropout,
                            dropatt=cfg.dropatt, tie_weight=True, d_embed=cfg.d_embed, div_val=cfg.dec_div_val,
                            tie_projs=self.tie_projs, pre_lnorm=True, tgt_len=cfg.dec_tgt_len, ext_len=cfg.dec_ext_len,
                            mem_len=cfg.dec_mem_len, cutoffs=self.cutoffs, attn_type=cfg.attn_type)
 
        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)
        
        

    def forward(self, z, *mems):
        
        src = self.embedding(z)
        src = src + z
        
        out, mems = self.decoder(src, src, *mems)
            
        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        
        return out_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class RandomSampling(nn.Module):
    def __init__(self, cfg):
        super(RandomSampling, self).__init__()
        
        self.mean = nn.Linear(cfg.dim_z, 1)
        self.logvar = nn.Linear(cfg.dim_z, 1)
        
        self.d_model = cfg.d_model
    
    def forward(self, z):
        # z.shape: (1, dim_feedforward, dim_z)
        z = z.squeeze(0)
        
        z_mean = self.mean(z).cuda()
        z_logvar = self.logvar(z).cuda()
        z_std = torch.exp(0.5 * z_logvar)
        
        rand_z = torch.randn(z.shape[0], self.d_model).cuda()
        
        result = rand_z * z_std + z_mean
        
        result = result.unsqueeze(0)
        
        KLD_loss = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1))
        
        return result, KLD_loss


class CADTransformer_VAE(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer_VAE, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)
        
        self.mems = tuple()
        
        self.bsz = cfg.batch_size
        
        self.random_sampling = RandomSampling(cfg)

    def forward(self, commands_enc, args_enc, z=None, mems=None,
                    return_tgt=True, encode_mode=False, encode_mem_mode=False):
           
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        
        if mems is not None:
            self.mems = mems

        if z is None:
            z, self.mems = self.encoder(commands_enc_, args_enc_, *self.mems)
            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        z, KLD_loss = self.random_sampling(z)
        
        if encode_mode: return _make_batch_first(z)
        if encode_mem_mode: return self.mems

        out_logits = self.decoder(z, *self.mems)
        out_logits = _make_batch_first(*out_logits)
        #if encode_mem_mode: return self.mems_dec
        
        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1],
            "KLD_loss": KLD_loss
        }

        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res
