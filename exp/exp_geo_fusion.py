import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

# 你的 Fusion 模型
from models.model_geo_fusion import Model_Geo_Fusion as FusionModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


class Exp_Geo_Fusion(Exp_Basic):
    """
    DualSG-Geo-Fusion 实验类

    支持 Geo_Fusion dataloader 输出（8项）：
    (
      batch_x, batch_y,
      batch_x_mark, batch_y_mark,
      element_id,
      group_id,
      norm,
      caption_emb
    )
    """

    def __init__(self, args):
        super().__init__(args)

    # ============================================================
    # 构建 Fusion 模型
    # ============================================================
    def _build_model(self, args):
        """
        FusionModel 内部通常会：
        - 加载一个数值流 ckpt（可选）
        - 冻结/复用数值流部件
        - 仅训练融合相关模块（依赖 requires_grad）
        """
        num_ckpt = getattr(args, "num_model_path", None)
        self.log(f"num_ckpt: {num_ckpt}")

        model = FusionModel(args, numerical_ckpt_path=num_ckpt).float()

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model

    # ============================================================
    # 数据加载
    # ============================================================
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples (Geo_Fusion)")
        return data_set, data_loader

    # ============================================================
    # Optimizer / Criterion
    # ============================================================
    def _select_optimizer(self):
        # 只优化 requires_grad=True 的参数（冻结部分不参与训练）
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate
        )

    def _select_criterion(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        if self.args.loss == "MAE":
            return nn.L1Loss()
        raise ValueError(f"Unsupported loss type: {self.args.loss}")

    # ============================================================
    # batch 解包
    # ============================================================
    def _unpack_batch(self, batch):
        """
        Geo_Fusion dataloader 必须返回 8 项：
        (x, y, x_mark, y_mark, element_id, group_id, norm, caption_emb)

        返回的 tensor 全部迁移到 self.device，并做 dtype / shape 规范化：
        - batch_x/batch_y: [B, L, 1]
        - x_mark/y_mark:  [L, 1] 或 [B, L, 1]
        - element_id/group_id: [B]
        - norm: [B, 3]
        - caption_emb: [B, D]
        """
        if len(batch) != 8:
            raise ValueError(
                f"Geo_Fusion expects 8 items "
                f"(x,y,x_mark,y_mark,element_id,group_id,norm,caption_emb), got {len(batch)}"
            )

        batch_x, batch_y, batch_x_mark, batch_y_mark, element_id, group_id, norm, caption_emb = batch

        # ---- 维度修正：保证 [B,L,1] ----
        # 有些 DataLoader collate 后可能是 [B,L]，这里兜底
        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(-1)
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(-1)

        # ---- to device ----
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        element_id = element_id.long().to(self.device)
        group_id = group_id.long().to(self.device)

        # norm: [B,3] -> float
        norm = norm.float().to(self.device)

        # caption_emb: [B,D] -> float
        caption_emb = caption_emb.float().to(self.device)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, element_id, group_id, norm, caption_emb

    # ============================================================
    # 构造 decoder 输入：与 Num 实验对齐（dec_inp）
    # ============================================================
    def _build_dec_inp(self, batch_y):
        """
        与你 Geo_Num_Meta exp 一致：
        dec_inp = [label_len 的真实 y] + [pred_len 的 zeros]
        """
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp],
            dim=1
        ).to(self.device)
        return dec_inp

    # ============================================================
    # Validation
    # ============================================================
    def vali(self, vali_loader, criterion):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in vali_loader:
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    element_id, group_id, norm, caption_emb
                ) = self._unpack_batch(batch)

                dec_inp = self._build_dec_inp(batch_y)

                # 统一 forward：与 Num 对齐，只多传 caption_emb
                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_id,
                    group_ids=group_id,
                    norms=norm,
                    caption_emb=caption_emb
                )

                # 监督目标：预测 pred_len
                f_dim = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(pred, target)
                losses.append(loss.item())

        self.model.train()
        return float(np.mean(losses))

    # ============================================================
    # Train
    # ============================================================
    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        _, test_loader = self._get_data("test")

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # ===== 打印可训练参数 =====
        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            epoch_time = time.time()

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}",
                unit="it"
            )

            for _, batch in pbar:
                model_optim.zero_grad()

                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    element_id, group_id, norm, caption_emb
                ) = self._unpack_batch(batch)

                dec_inp = self._build_dec_inp(batch_y)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_id,
                    group_ids=group_id,
                    norms=norm,
                    caption_emb=caption_emb
                )

                f_dim = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                # ===== 主loss =====
                loss_main = criterion(pred, target)

                # ===== aux_loss（纠偏正则，可选）=====
                m = self.model.module if hasattr(self.model, "module") else self.model
                aux_loss = getattr(m, "aux_loss", None)

                loss = loss_main
                if aux_loss is not None:
                    loss = loss + aux_loss

                train_losses.append(loss.item())

                loss.backward()
                model_optim.step()


            train_loss = float(np.mean(train_losses))
            vali_loss = self.vali(vali_loader, criterion)

            self.log(
                f"Epoch {epoch + 1} | "
                f"Train {train_loss:.6f} | "
                f"Val {vali_loss:.6f} | "
                f"Time {time.time() - epoch_time:.2f}s"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.log("Early stopping")
                break

            if getattr(self.args, "adjust", 0):
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(
            torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu")
        )
        return self.model

    # ============================================================
    # Test
    # ============================================================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")

        if test:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoint_dir, setting, "checkpoint.pth"),
                    map_location="cpu"
                )
            )

        self.model.eval()
        preds, trues = [], []

        # 输出目录
        result_folder = os.path.join(self.args.output_dir, setting, "results/")
        os.makedirs(result_folder, exist_ok=True)

        vis_folder = os.path.join(self.args.output_dir, setting, "visualizations/")
        if self.args.visualize:
            os.makedirs(vis_folder, exist_ok=True)

        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    element_id, group_id, norm, caption_emb
                ) = self._unpack_batch(batch)

                dec_inp = self._build_dec_inp(batch_y)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_id,
                    group_ids=group_id,
                    norms=norm,
                    caption_emb=caption_emb
                )

                # 截取 pred_len
                f_dim = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len:, f_dim:]
                true = batch_y[:, -self.args.pred_len:, f_dim:]

                # =====================================================
                # GeoStyle
                # norm: [B,3] -> min/max/eps
                # =====================================================
                min_v = norm[:, 0].view(-1, 1, 1)
                max_v = norm[:, 1].view(-1, 1, 1)

                pred_denorm = pred * (max_v - min_v) + min_v
                true_denorm = true * (max_v - min_v) + min_v

                preds.append(pred_denorm.detach().cpu().numpy())
                trues.append(true_denorm.detach().cpu().numpy())

                # =====================================================
                # PDF 可视化
                # =====================================================
                if self.args.visualize and i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()

                    min_v_np = min_v.detach().cpu().numpy()
                    max_v_np = max_v.detach().cpu().numpy()
                    input_denorm = input_x * (max_v_np - min_v_np) + min_v_np

                    pred_np = pred_denorm.detach().cpu().numpy()
                    true_np = true_denorm.detach().cpu().numpy()

                    gt = np.concatenate((input_denorm[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_denorm[0, :, -1], pred_np[0, :, -1]), axis=0)

                    T = input_denorm.shape[1]
                    from utils.tools import visual
                    visual(gt, pd, os.path.join(vis_folder, f"{i}.pdf"), history_len=T)

        self.log(f"Inference time: {time.time() - start_time:.2f}s")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)

        self.log("\n" + "=" * 50)
        self.log("Test Results (GeoStyle Fusion)")
        self.log("=" * 50)
        self.log(f"MAE:  {mae:.6f}")
        self.log(f"MSE:  {mse:.6f}")
        self.log(f"RMSE: {rmse:.6f}")
        self.log(f"MAPE: {mape * 100:.2f}%")
        self.log(f"MSPE: {mspe:.6f}")
        self.log(f"WAPE: {wape * 100:.2f}%")
        self.log("=" * 50)

        np.save(result_folder + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, wape]))
        np.save(result_folder + "pred.npy", preds)
        np.save(result_folder + "true.npy", trues)

        # 汇总 PDF
        self._plot_predictions(
            preds, trues,
            result_folder, setting,
            mae=mae, mse=mse, rmse=rmse, mape=mape, wape=wape
        )

        return mae, mse, rmse, mape, mspe, wape

    # ============================================================
    # 汇总可视化：predictions.pdf
    # ============================================================
    def _plot_predictions(self, preds, trues, folder_path, setting, mae, mse, rmse, mape, wape, num_samples=10):
        """
        输出综合 PDF：包含散点、误差分布、逐步 MAE、指标表，以及若干样本曲线。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        reports_folder = os.path.join(self.args.output_dir, setting, "reports/")
        os.makedirs(reports_folder, exist_ok=True)

        pdf_path = os.path.join(reports_folder, "predictions.pdf")

        np.random.seed(self.args.seed)
        total_samples = preds.shape[0]
        sample_indices = np.random.choice(
            total_samples,
            min(num_samples, total_samples),
            replace=False
        )

        with PdfPages(pdf_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"Model_Geo_Fusion Prediction Results\n{setting}",
                fontsize=14
            )

            ax = axes[0, 0]
            ax.scatter(trues.flatten()[::100], preds.flatten()[::100], alpha=0.3, s=1)
            ax.plot([trues.min(), trues.max()], [trues.min(), trues.max()], "r--", lw=2)
            ax.set_title("Prediction vs Ground Truth")

            ax = axes[0, 1]
            errors = preds.flatten() - trues.flatten()
            ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="r", linestyle="--")
            ax.set_title(f"Error Distribution (Mean: {errors.mean():.4f})")

            ax = axes[1, 0]
            mae_per_step = np.mean(np.abs(preds - trues), axis=0).squeeze()
            ax.bar(range(len(mae_per_step)), mae_per_step)
            ax.set_title("MAE per Prediction Step")

            ax = axes[1, 1]
            ax.axis("off")
            table = ax.table(
                cellText=[
                    ["MAE", f"{mae:.6f}"],
                    ["MSE", f"{mse:.6f}"],
                    ["RMSE", f"{rmse:.6f}"],
                    ["MAPE", f"{mape * 100:.2f}%"],
                    ["WAPE", f"{wape * 100:.2f}%"],
                ],
                colLabels=["Metric", "Value"],
                loc="center",
                cellLoc="center"
            )
            table.scale(1.2, 1.5)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # 若干样本曲线
            for idx in sample_indices:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(trues[idx].squeeze(), label="GT")
                ax.plot(preds[idx].squeeze(), label="Pred")
                ax.legend()
                ax.set_title(f"Sample {idx}")
                pdf.savefig(fig)
                plt.close()

        self.log(f"✅ Predictions PDF saved to: {pdf_path}")
