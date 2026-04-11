import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.masking import TriangularCausalMask, ProbMask


class FlattenHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear = nn.LazyLinear(output_size)

    def forward(self, x):
        return self.linear(x)


class Model_Fit_Num(nn.Module):
    def __init__(self, configs):
        super(Model_Fit_Num, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq = configs.freq

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=getattr(configs, 'output_attention', False)
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True, configs.factor, attention_dropout=configs.dropout, output_attention=getattr(configs, 'output_attention', False)
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=getattr(configs, 'output_attention', False)
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.LazyLinear(configs.c_out)
        )

        # 预测头
        self.flatten_head = FlattenHead(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, enc_in]
        # x_mark_enc: [batch_size, seq_len, 4] or [seq_len, 1]
        # x_dec: [batch_size, label_len+pred_len, dec_in]
        # x_mark_dec: [batch_size, label_len+pred_len, 4] or [label_len+pred_len, 1]

        # 处理时间戳数据形状
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]
        dec_seq_len = x_dec.shape[1]
        
        # 确保x_mark_enc和x_mark_dec有batch维度
        if x_mark_enc.dim() == 2:
            x_mark_enc = x_mark_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        if x_mark_dec.dim() == 2:
            x_mark_dec = x_mark_dec.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 确保时间戳数据长度与输入序列长度匹配
        if x_mark_enc.shape[1] != seq_len:
            # 如果时间戳长度不匹配，使用正确的长度重新生成
            x_mark_enc = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1).float().to(x_enc.device) / seq_len
        if x_mark_dec.shape[1] != dec_seq_len:
            # 如果时间戳长度不匹配，使用正确的长度重新生成
            x_mark_dec = torch.arange(dec_seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1).float().to(x_enc.device) / self.pred_len
        
        # 对于TimeFeatureEmbedding，我们需要生成正确维度的时间特征
        # 由于我们的时间戳只是简单的序列位置，我们需要转换为TimeFeatureEmbedding期望的格式
        # 对于'w'频率，TimeFeatureEmbedding期望2维输入
        if x_mark_enc.shape[-1] == 1:
            # 生成额外的时间特征
            seq_len = x_mark_enc.shape[1]
            # 使用配置中的序列长度作为周期长度
            period = self.seq_len
            # 添加一个额外的维度作为周期信息
            extra_feature = (torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1).float() / period).to(x_enc.device)
            x_mark_enc = torch.cat([x_mark_enc, extra_feature], dim=-1)
        if x_mark_dec.shape[-1] == 1:
            # 生成额外的时间特征
            dec_seq_len = x_mark_dec.shape[1]
            # 使用配置中的序列长度作为周期长度
            period = self.seq_len
            extra_feature = (torch.arange(dec_seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1).float() / period).to(x_enc.device)
            x_mark_dec = torch.cat([x_mark_dec, extra_feature], dim=-1)

        # 编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 解码
        dec_out = self.enc_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # 预测
        output = self.flatten_head(dec_out[:, -self.pred_len:, :])

        return output  # [batch_size, pred_len, c_out]