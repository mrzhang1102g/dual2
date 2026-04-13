"""
FIT 融合流实验。

保持现有融合思路不变：
- 数值主干来自 Model_Fit_Num_With_Meta
- 文本输入仍是离线 caption embedding
- 支持 direct / residual 两种融合方式
"""

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
from models.model_fit_fusion import Model_Fit_Fusion as FusionModel
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


class Exp_Fit_Fusion(Exp_Basic):
    """FIT 融合流实验类。"""

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self, args):
        num_ckpt = getattr(args, "num_model_path", None)
        self.log(f"num_ckpt: {num_ckpt}")

        model = FusionModel(args, numerical_ckpt_path=num_ckpt).float()
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples (FIT Fusion)")
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_mode = getattr(self.args, "fusion_optimizer_mode", "unified")
        if optimizer_mode == "unified":
            trainable_params = [
                param for param in self.model.parameters() if param.requires_grad
            ]
            if not trainable_params:
                raise ValueError("No trainable parameters found for FIT fusion optimizer.")
            return optim.Adam(
                [
                    {
                        "params": trainable_params,
                        "lr": self.args.learning_rate,
                        "base_lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                        "group_name": "all",
                    }
                ]
            )

        if optimizer_mode != "split":
            raise ValueError(f"Unsupported fusion_optimizer_mode: {optimizer_mode}")

        module = self.model.module if hasattr(self.model, "module") else self.model
        numerical_param_ids = {
            id(param)
            for param in module.numerical_model.parameters()
            if param.requires_grad
        }

        numerical_params = []
        text_fusion_params = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in numerical_param_ids:
                numerical_params.append(param)
            else:
                text_fusion_params.append(param)

        param_groups = []
        if numerical_params:
            param_groups.append(
                {
                    "params": numerical_params,
                    "lr": self.args.lr_num,
                    "base_lr": self.args.lr_num,
                    "weight_decay": self.args.weight_decay,
                    "group_name": "numerical",
                }
            )
        if text_fusion_params:
            param_groups.append(
                {
                    "params": text_fusion_params,
                    "lr": self.args.lr_text,
                    "base_lr": self.args.lr_text,
                    "weight_decay": self.args.weight_decay_text,
                    "group_name": "text_fusion",
                }
            )

        if not param_groups:
            raise ValueError("No trainable parameters found for FIT fusion optimizer.")

        return optim.Adam(param_groups)

    def _select_criterion(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        if self.args.loss == "MAE":
            return nn.L1Loss()
        raise ValueError(f"Unsupported loss type: {self.args.loss}")

    def _prepare_batch(self, batch):
        (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            city_id,
            gender_id,
            age_id,
            element_id,
            caption_emb,
        ) = batch

        # FIT fusion 的 loader 本来就应返回 [B, L, 1]，
        # 这里保留一个轻量兜底，避免后续切 loader 时 batch 维度不一致。
        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(-1)
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(-1)

        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
            city_id.long().to(self.device),
            gender_id.long().to(self.device),
            age_id.long().to(self.device),
            element_id.long().to(self.device),
            caption_emb.float().to(self.device),
        )

    def _forward_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, city_id, gender_id, age_id, element_id, caption_emb = batch
        outputs = self.model(
            batch_x,
            batch_x_mark,
            batch_y,
            batch_y_mark,
            city_id,
            gender_id,
            age_id,
            element_id,
            caption_emb,
        )
        target = batch_y[:, -self.args.pred_len :, -1:]
        return outputs, target

    def _compute_loss(self, outputs, target, criterion):
        loss = criterion(outputs, target)
        module = self.model.module if hasattr(self.model, "module") else self.model
        aux_loss = getattr(module, "aux_loss", None)
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss

    def _warmup_model(self, train_loader):
        """
        某些 LazyLinear 需要先过一次前向，训练前在这里完成初始化。
        """
        for _, batch in enumerate(train_loader):
            batch = self._prepare_batch(batch)
            with torch.no_grad():
                self._forward_batch(batch)
            break

    def _save_test_outputs(self, setting, preds, trues, metrics_tuple):
        result_folder = os.path.join(self.args.output_dir, setting, "results")
        os.makedirs(result_folder, exist_ok=True)

        mae, mse, rmse, mape, mspe, wape = metrics_tuple
        with open(os.path.join(result_folder, "result.txt"), "w", encoding="utf-8") as file:
            file.write(setting + "  \n")
            file.write(
                f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, "
                f"MAPE: {mape * 100:.2f}%, WAPE: {wape * 100:.2f}%"
            )
            file.write("\n\n")

        np.save(os.path.join(result_folder, "metrics.npy"), np.array(metrics_tuple, dtype=np.float32))
        np.save(os.path.join(result_folder, "pred.npy"), preds)
        np.save(os.path.join(result_folder, "true.npy"), trues)

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        _, _ = self._get_data("test")

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

        # 先 warmup，再建 optimizer，避免 LazyLinear 还没初始化就被优化器捕获。
        self._warmup_model(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            epoch_time = time.time()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="it")
            for _, batch in pbar:
                model_optim.zero_grad()
                batch = self._prepare_batch(batch)
                outputs, target = self._forward_batch(batch)
                loss = self._compute_loss(outputs, target, criterion)
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

            if self.args.adjust:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu"))
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in vali_loader:
                batch = self._prepare_batch(batch)
                outputs, target = self._forward_batch(batch)
                losses.append(self._compute_loss(outputs, target, criterion).item())

        self.model.train()
        return float(np.mean(losses))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")
        if test:
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoint_dir, setting, "checkpoint.pth"), map_location="cpu")
            )

        self.model.eval()
        preds, trues = [], []

        max_n = int(getattr(self.args, "max_test_samples", -1) or -1)
        seen = 0

        vis_folder = os.path.join(self.args.output_dir, setting, "visualizations")
        if self.args.visualize:
            os.makedirs(vis_folder, exist_ok=True)
            self.log(f"[TEST] Batch-level PDFs will be saved to: {vis_folder}")
        if max_n > 0:
            self.log(f"[TEST] Will evaluate only first {max_n} test samples.")

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = self._prepare_batch(batch)
                batch_x, batch_y = batch[0], batch[1]
                outputs, target = self._forward_batch(batch)

                pred_np = outputs.detach().cpu().numpy()
                true_np = target.detach().cpu().numpy()

                if max_n > 0:
                    remain = max_n - seen
                    if remain <= 0:
                        break
                    if pred_np.shape[0] > remain:
                        pred_np = pred_np[:remain]
                        true_np = true_np[:remain]

                preds.append(pred_np)
                trues.append(true_np)
                seen += pred_np.shape[0]

                if self.args.visualize and i % 20 == 0 and pred_np.shape[0] > 0:
                    keep_b = pred_np.shape[0]
                    input_x = batch_x.detach().cpu().numpy()[:keep_b]
                    pred_vis = outputs.detach().cpu().numpy()[:keep_b]
                    true_vis = target.detach().cpu().numpy()[:keep_b]

                    if self.args.inverse:
                        input_x = test_data.inverse_transform(input_x)
                        pred_vis = test_data.inverse_transform(pred_vis)
                        true_vis = test_data.inverse_transform(true_vis)

                    gt = np.concatenate((input_x[0, :, -1], true_vis[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], pred_vis[0, :, -1]), axis=0)
                    pdf_path = os.path.join(vis_folder, f"{i}.pdf")
                    visual(gt, pd, pdf_path)
                    self.log(f"[TEST][PDF] Saved batch-level PDF: {pdf_path}")

                if max_n > 0 and seen >= max_n:
                    break

        preds = np.concatenate(preds, axis=0) if preds else np.array([])
        trues = np.concatenate(trues, axis=0) if trues else np.array([])

        if self.args.inverse and preds.size > 0:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        metrics_tuple = metric(preds, trues)
        mae, mse, rmse, mape, mspe, wape = metrics_tuple

        self.log("\n" + "=" * 50)
        self.log("Test Results (FIT)")
        self.log("=" * 50)
        self.log(f"MAE:  {mae:.6f}")
        self.log(f"MSE:  {mse:.6f}")
        self.log(f"RMSE: {rmse:.6f}")
        self.log(f"MAPE: {mape * 100:.2f}%")
        self.log(f"MSPE: {mspe:.6f}")
        self.log(f"WAPE: {wape * 100:.2f}%")
        self.log("=" * 50)

        self._save_test_outputs(setting, preds, trues, metrics_tuple)

        if self.args.visualize and preds.size > 0:
            self._plot_predictions(preds, trues, setting, mae, mse, rmse, mape, wape)
            self.log(f"[TEST][PDF] Saved summary PDF: {os.path.join(self.args.output_dir, setting, 'reports', 'predictions.pdf')}")

        return metrics_tuple

    def _plot_predictions(self, preds, trues, setting, mae, mse, rmse, mape, wape, num_samples=10):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        reports_folder = os.path.join(self.args.output_dir, setting, "reports")
        os.makedirs(reports_folder, exist_ok=True)
        pdf_path = os.path.join(reports_folder, "predictions.pdf")

        np.random.seed(self.args.seed)
        total_samples = preds.shape[0]
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

        with PdfPages(pdf_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Model_Fit_Fusion Prediction Results\n{setting}", fontsize=14)

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
                cellLoc="center",
            )
            table.scale(1.2, 1.5)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            for index in sample_indices:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(trues[index].squeeze(), label="GT")
                ax.plot(preds[index].squeeze(), label="Pred")
                ax.legend()
                ax.set_title(f"Sample {index}")
                pdf.savefig(fig)
                plt.close()

        self.log(f"Predictions PDF saved to: {pdf_path}")
