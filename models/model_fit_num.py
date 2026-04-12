import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer


class FlattenHead(nn.Module):
    """对 decoder 的最后一维做线性映射，输出预测通道数。"""

    def __init__(self, output_size):
        super().__init__()
        self.linear = nn.LazyLinear(output_size)

    def forward(self, x):
        return self.linear(x)


class Model_Fit_Num(nn.Module):
    """
    FIT 纯数值流基线模型。

    整体仍然保持原来的 encoder-decoder 结构，只把时间标记的兜底逻辑拆成辅助函数，
    这样后续排查输入 shape 问题时更容易定位。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq = configs.freq
        self.output_attention = getattr(configs, "output_attention", False)

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.LazyLinear(configs.c_out),
        )

        self.flatten_head = FlattenHead(configs.c_out)

    @staticmethod
    def _ensure_batch_axis(x_mark, batch_size):
        if x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(0).repeat(batch_size, 1, 1)
        return x_mark

    @staticmethod
    def _build_position_mark(batch_size, length, denominator, device):
        positions = torch.arange(length, device=device).float().view(1, length, 1)
        return positions.repeat(batch_size, 1, 1) / float(denominator)

    def _prepare_time_mark(self, x_mark, batch_size, length, device, denominator):
        """
        当前 FIT loader 只提供简单位置特征。
        这里保留旧逻辑：
        1. 若 batch 维缺失，则补成 [B, L, D]
        2. 若长度不匹配，则按当前位置重建
        3. 若最后一维只有 1，则补一维归一化周期特征，适配 TimeFeatureEmbedding
        """
        x_mark = self._ensure_batch_axis(x_mark, batch_size)
        if x_mark.shape[1] != length:
            x_mark = self._build_position_mark(batch_size, length, denominator, device)

        if x_mark.shape[-1] == 1:
            extra_feature = self._build_position_mark(batch_size, length, max(self.seq_len, 1), device)
            x_mark = torch.cat([x_mark, extra_feature], dim=-1)

        return x_mark

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        enc_len = x_enc.shape[1]
        dec_len = x_dec.shape[1]

        x_mark_enc = self._prepare_time_mark(
            x_mark=x_mark_enc,
            batch_size=batch_size,
            length=enc_len,
            device=x_enc.device,
            denominator=max(enc_len, 1),
        )
        x_mark_dec = self._prepare_time_mark(
            x_mark=x_mark_dec,
            batch_size=batch_size,
            length=dec_len,
            device=x_enc.device,
            denominator=max(self.pred_len, 1),
        )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.enc_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # 这里只对 decoder 最后 pred_len 段做投影，保持原来的输出语义不变。
        return self.flatten_head(dec_out[:, -self.pred_len :, :])
