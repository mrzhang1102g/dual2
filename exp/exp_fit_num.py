"""
FIT 数值流实验。

本文件只负责两种模型：
- Model_Fit_Num
- Model_Fit_Num_With_Meta

不再混入 Geo / Fusion 的分支判断。
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
from utils.dtw_metric import accelerated_dtw
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


class Exp_Fit_Num(Exp_Basic):
    """FIT 数值流实验类。"""

    SUPPORTED_MODELS = {"Model_Fit_Num", "Model_Fit_Num_With_Meta"}

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self, args):
        if args.model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Exp_Fit_Num 只支持 {sorted(self.SUPPORTED_MODELS)}，当前收到 {args.model}"
            )

        model = self.model_dict[args.model](args).float()
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples (FIT Num)")
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        return nn.L1Loss()

    def _warmup_model(self, train_loader):
        """
        数值模型里也有 LazyLinear，先用一个 batch 跑通前向，避免本地 smoke 时初始化时机不稳定。
        """
        for _, batch in enumerate(train_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._unpack_batch(batch)
            batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._move_batch_to_device(
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs
            )
            with torch.no_grad():
                self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs)
            break

    def _use_meta_model(self):
        return self.args.model == "Model_Fit_Num_With_Meta"

    def _unpack_batch(self, batch):
        """
        loader 始终返回：
        (x, y, x_mark, y_mark, city_id, gender_id, age_id, element_id)

        纯数值模型只用前四项，meta 模型再额外消费后四项。
        """
        if len(batch) < 4:
            raise ValueError(f"FIT 数值流 batch 至少应包含 4 项，当前得到 {len(batch)} 项")

        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
        meta_inputs = {}
        if self._use_meta_model():
            if len(batch) < 8:
                raise ValueError("Model_Fit_Num_With_Meta 需要 city/gender/age/element 四个元数据输入")
            meta_inputs = {
                "city_ids": batch[4],
                "gender_ids": batch[5],
                "age_ids": batch[6],
                "element_ids": batch[7],
            }
        return batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs

    def _move_batch_to_device(self, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        device_meta = {}
        for key, value in meta_inputs.items():
            device_meta[key] = value.long().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, device_meta

    def _build_dec_inp(self, batch_y):
        zeros = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        return torch.cat([batch_y[:, : self.args.label_len, :], zeros], dim=1).to(self.device)

    def _forward_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs):
        dec_inp = self._build_dec_inp(batch_y)
        model_kwargs = meta_inputs if self._use_meta_model() else {}

        if self.args.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **model_kwargs)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **model_kwargs)

        f_dim = -1 if self.args.features == "MS" else 0
        preds = outputs[:, -self.args.pred_len :, f_dim:]
        trues = batch_y[:, -self.args.pred_len :, f_dim:]
        return preds, trues

    def vali(self, vali_data, vali_loader, criterion, desc="Validation"):
        self.model.eval()
        losses = []

        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc=desc, unit="it")
            for _, batch in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._unpack_batch(batch)
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs
                )
                preds, trues = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs)
                losses.append(criterion(preds, trues).item())

        self.model.train()
        return float(np.mean(losses))

    def train(self, setting):
        _, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        _, _ = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self._warmup_model(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        use_amp = self.args.use_amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []
            epoch_time = time.time()
            self.model.train()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="it")
            for i, batch in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._unpack_batch(batch)
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs
                )

                iter_count += 1
                model_optim.zero_grad()
                preds, trues = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs)
                loss = criterion(preds, trues)
                train_losses.append(loss.item())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    self.log(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.log(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            train_loss = float(np.mean(train_losses))
            vali_loss = self.vali(vali_data, vali_loader, criterion, desc="Validation")

            self.log(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            self.log(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.log("Early stopping")
                break

            if self.args.adjust:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu"))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        preds, trues = [], []
        results_folder = os.path.join(self.args.output_dir, setting, "results/")
        os.makedirs(results_folder, exist_ok=True)

        vis_folder = os.path.join(self.args.output_dir, setting, "visualizations/")
        if self.args.visualize:
            os.makedirs(vis_folder, exist_ok=True)

        max_n = int(getattr(self.args, "max_test_samples", -1) or -1)
        seen = 0

        self.model.eval()
        start_time_test = time.time()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", unit="it")
            for i, batch in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._unpack_batch(batch)
                batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs
                )
                pred_tensor, true_tensor = self._forward_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs)

                pred_np = pred_tensor.detach().cpu().numpy()
                true_np = true_tensor.detach().cpu().numpy()

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
                    input_np = batch_x.detach().cpu().numpy()
                    keep_b = pred_np.shape[0]
                    input_np = input_np[:keep_b]

                    if test_data.scale and self.args.inverse:
                        input_np = test_data.inverse_transform(input_np)
                        pred_vis = test_data.inverse_transform(pred_np)
                        true_vis = test_data.inverse_transform(true_np)
                    else:
                        pred_vis = pred_np
                        true_vis = true_np

                    gt = np.concatenate((input_np[0, :, -1], true_vis[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred_vis[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(vis_folder, f"{i}.pdf"))

                if max_n > 0 and seen >= max_n:
                    break

        self.log(f"Inference time: {time.time() - start_time_test:.2f} seconds")

        preds = np.concatenate(preds, axis=0) if preds else np.array([])
        trues = np.concatenate(trues, axis=0) if trues else np.array([])

        if preds.size > 0 and test_data.scale and self.args.inverse:
            if preds.shape[-1] != trues.shape[-1] and trues.shape[-1] % preds.shape[-1] == 0:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        self.log(f"test shape: {preds.shape} {trues.shape}")

        if self.args.use_dtw and preds.size > 0:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for index in range(preds.shape[0]):
                if index % 100 == 0:
                    self.log(f"calculating dtw iter: {index}")
                distance, _, _, _ = accelerated_dtw(
                    preds[index].reshape(-1, 1),
                    trues[index].reshape(-1, 1),
                    dist=manhattan_distance,
                )
                dtw_list.append(distance)
            dtw_value = float(np.mean(dtw_list))
        else:
            dtw_value = "Not calculated"

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)

        self.log("\n" + "=" * 50)
        self.log("Test Results:")
        self.log("=" * 50)
        self.log(f"MAE:  {mae:.6f}")
        self.log(f"MSE:  {mse:.6f}")
        self.log(f"RMSE: {rmse:.6f}")
        self.log(f"MAPE: {mape * 100:.2f}%")
        self.log(f"MSPE: {mspe:.6f}")
        self.log(f"WAPE: {wape * 100:.2f}%")
        if self.args.use_dtw and isinstance(dtw_value, float):
            self.log(f"DTW:  {dtw_value:.6f}")
        self.log("=" * 50 + "\n")

        with open(os.path.join(results_folder, "result.txt"), "w", encoding="utf-8") as file:
            file.write(setting + "  \n")
            file.write(
                f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, "
                f"MAPE: {mape * 100:.2f}%, WAPE: {wape * 100:.2f}%"
            )
            if self.args.use_dtw and isinstance(dtw_value, float):
                file.write(f", DTW: {dtw_value:.6f}")
            file.write("\n\n")

        np.save(results_folder + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, wape], dtype=np.float32))
        np.save(results_folder + "pred.npy", preds)
        np.save(results_folder + "true.npy", trues)

        if self.args.visualize and preds.size > 0:
            self._plot_predictions(preds, trues, setting)

        return mae, mse, rmse, mape, mspe, wape

    def _plot_predictions(self, preds, trues, setting, num_samples=10):
        """生成 FIT 数值流的预测结果 PDF。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_folder = os.path.join(self.args.output_dir, setting, "reports/")
        os.makedirs(pdf_folder, exist_ok=True)

        pdf_path = os.path.join(pdf_folder, "predictions.pdf")
        np.random.seed(self.args.seed)
        total_samples = preds.shape[0]
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

        with PdfPages(pdf_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Model_Fit_Num Prediction Results\n{setting}", fontsize=14)

            ax = axes[0, 0]
            pred_flat = preds.flatten()
            true_flat = trues.flatten()
            ax.scatter(true_flat[::100], pred_flat[::100], alpha=0.3, s=1)
            ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], "r--", lw=2)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Prediction vs Ground Truth")

            ax = axes[0, 1]
            errors = pred_flat - true_flat
            ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="r", linestyle="--")
            ax.set_xlabel("Prediction Error")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Error Distribution (Mean: {errors.mean():.4f})")

            ax = axes[1, 0]
            mae_per_step = np.mean(np.abs(preds - trues), axis=0).squeeze()
            ax.bar(range(len(mae_per_step)), mae_per_step, color="steelblue")
            ax.set_xlabel("Prediction Step (Week)")
            ax.set_ylabel("MAE")
            ax.set_title("MAE per Prediction Step")

            ax = axes[1, 1]
            ax.axis("off")
            mae, mse, rmse, mape, _, wape = metric(preds, trues)
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
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax.set_title("Metrics Summary")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            for index in sample_indices:
                fig, ax = plt.subplots(figsize=(12, 4))
                pred_seq = preds[index].squeeze()
                true_seq = trues[index].squeeze()
                weeks = range(1, len(pred_seq) + 1)
                ax.plot(weeks, true_seq, "b-", label="Ground Truth", linewidth=2, marker="o", markersize=4)
                ax.plot(weeks, pred_seq, "r--", label="Prediction", linewidth=2, marker="s", markersize=4)
                ax.fill_between(weeks, true_seq, pred_seq, alpha=0.3, color="gray")
                sample_mae = np.mean(np.abs(pred_seq - true_seq))
                ax.set_xlabel("Week")
                ax.set_ylabel("Trend Value")
                ax.set_title(f"Sample {index} - Prediction (MAE: {sample_mae:.4f})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

        self.log(f"Predictions PDF saved to: {pdf_path}")
