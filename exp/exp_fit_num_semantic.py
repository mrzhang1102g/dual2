"""FIT numeric+meta experiment with structured-text semantic supervision."""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.dtw_metric import accelerated_dtw
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


class Exp_Fit_Num_Semantic(Exp_Basic):
    """FIT numeric+meta experiment with semantic multitask supervision."""

    SUPPORTED_MODELS = {"Model_Fit_Num_With_Meta_Semantic"}

    def __init__(self, args):
        super().__init__(args)
        self.forecast_criterion = self._select_forecast_criterion()
        self.semantic_criterion = nn.CrossEntropyLoss(reduction="none")

    def _build_model(self, args):
        if args.model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Exp_Fit_Num_Semantic only supports {sorted(self.SUPPORTED_MODELS)}, got {args.model}"
            )

        model = self.model_dict[args.model](args).float()
        pretrained_path = getattr(args, "pretrained_num_model_path", "")
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            model_state = model.state_dict()
            matched_state = {}
            for key, value in checkpoint.items():
                if key not in model_state:
                    continue

                target_param = model_state[key]
                if isinstance(target_param, (UninitializedParameter, UninitializedBuffer)):
                    matched_state[key] = value
                    continue

                if tuple(target_param.shape) == tuple(value.shape):
                    matched_state[key] = value

            model_state.update(matched_state)
            model.load_state_dict(model_state)
            self.log(
                f"[Semantic] loaded {len(matched_state)} matching parameters from pretrained numeric checkpoint: "
                f"{pretrained_path}"
            )
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples (FIT Meta Semantic)")
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_forecast_criterion(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        return nn.L1Loss()

    def _warmup_model(self, train_loader):
        for _, batch in enumerate(train_loader):
            unpacked = self._prepare_batch(batch)
            with torch.no_grad():
                self._forward_batch(*unpacked)
            break

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
        city_ids, gender_ids, age_ids, element_ids = batch[4:8]
        semantic_targets = batch[8]

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        meta_inputs = {
            "city_ids": city_ids.long().to(self.device),
            "gender_ids": gender_ids.long().to(self.device),
            "age_ids": age_ids.long().to(self.device),
            "element_ids": element_ids.long().to(self.device),
        }

        device_targets = {
            "overall_label": semantic_targets["overall_label"].long().to(self.device),
            "recent_label": semantic_targets["recent_label"].long().to(self.device),
            "volatility_label": semantic_targets["volatility_label"].long().to(self.device),
            "turning_label": semantic_targets["turning_label"].long().to(self.device),
            "overall_mask": semantic_targets["overall_mask"].float().to(self.device),
            "recent_mask": semantic_targets["recent_mask"].float().to(self.device),
            "volatility_mask": semantic_targets["volatility_mask"].float().to(self.device),
            "turning_mask": semantic_targets["turning_mask"].float().to(self.device),
        }
        return batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs, device_targets

    def _build_dec_inp(self, batch_y):
        zeros = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        return torch.cat([batch_y[:, : self.args.label_len, :], zeros], dim=1).to(self.device)

    def _forward_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_inputs, semantic_targets):
        dec_inp = self._build_dec_inp(batch_y)
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **meta_inputs)

        f_dim = -1 if self.args.features == "MS" else 0
        preds = outputs["forecast"][:, -self.args.pred_len :, f_dim:]
        trues = batch_y[:, -self.args.pred_len :, f_dim:]
        logits = outputs["semantic_logits"]

        forecast_loss = self.forecast_criterion(preds, trues)
        semantic_loss, semantic_breakdown = self._compute_semantic_loss(logits, semantic_targets)
        total_loss = forecast_loss + self.args.semantic_loss_weight * semantic_loss

        return {
            "preds": preds,
            "trues": trues,
            "forecast_loss": forecast_loss,
            "semantic_loss": semantic_loss,
            "total_loss": total_loss,
            "semantic_breakdown": semantic_breakdown,
        }

    def _masked_ce(self, logits, labels, masks):
        per_sample = self.semantic_criterion(logits, labels)
        denom = masks.sum()
        if denom.item() <= 0:
            return logits.new_tensor(0.0)
        return (per_sample * masks).sum() / denom

    def _compute_semantic_loss(self, logits, semantic_targets):
        breakdown = {
            "overall": self._masked_ce(
                logits["overall"],
                semantic_targets["overall_label"],
                semantic_targets["overall_mask"],
            ),
            "recent": self._masked_ce(
                logits["recent"],
                semantic_targets["recent_label"],
                semantic_targets["recent_mask"],
            ),
            "volatility": self._masked_ce(
                logits["volatility"],
                semantic_targets["volatility_label"],
                semantic_targets["volatility_mask"],
            ),
            "turning": self._masked_ce(
                logits["turning"],
                semantic_targets["turning_label"],
                semantic_targets["turning_mask"],
            ),
        }
        total = sum(breakdown.values()) / len(breakdown)
        return total, breakdown

    def vali(self, vali_data, vali_loader, desc="Validation"):
        del vali_data
        self.model.eval()
        total_losses, forecast_losses, semantic_losses = [], [], []

        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc=desc, unit="it")
            for _, batch in pbar:
                outputs = self._forward_batch(*self._prepare_batch(batch))
                total_losses.append(outputs["total_loss"].item())
                forecast_losses.append(outputs["forecast_loss"].item())
                semantic_losses.append(outputs["semantic_loss"].item())

        self.model.train()
        return {
            "total": float(np.mean(total_losses)),
            "forecast": float(np.mean(forecast_losses)),
            "semantic": float(np.mean(semantic_losses)),
        }

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
        use_amp = self.args.use_amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            total_losses, forecast_losses, semantic_losses = [], [], []
            epoch_time = time.time()
            self.model.train()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="it")
            for i, batch in pbar:
                prepared = self._prepare_batch(batch)
                iter_count += 1
                model_optim.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_batch(*prepared)
                        loss = outputs["total_loss"]
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_batch(*prepared)
                    loss = outputs["total_loss"]
                    loss.backward()
                    model_optim.step()

                total_losses.append(outputs["total_loss"].item())
                forecast_losses.append(outputs["forecast_loss"].item())
                semantic_losses.append(outputs["semantic_loss"].item())

                if (i + 1) % 100 == 0:
                    self.log(f"\titers: {i + 1}, epoch: {epoch + 1} | total: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.log(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            train_total = float(np.mean(total_losses))
            train_forecast = float(np.mean(forecast_losses))
            train_semantic = float(np.mean(semantic_losses))
            vali_metrics = self.vali(vali_data, vali_loader, desc="Validation")

            self.log(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            self.log(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Total: {train_total:.7f} Train Forecast: {train_forecast:.7f} "
                f"Train Semantic: {train_semantic:.7f} | "
                f"Val Total: {vali_metrics['total']:.7f} Val Forecast: {vali_metrics['forecast']:.7f} "
                f"Val Semantic: {vali_metrics['semantic']:.7f}"
            )

            early_stopping(vali_metrics["total"], self.model, path)
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
                outputs = self._forward_batch(*self._prepare_batch(batch))

                pred_np = outputs["preds"].detach().cpu().numpy()
                true_np = outputs["trues"].detach().cpu().numpy()

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
                    batch_x = batch[0].detach().cpu().numpy()
                    keep_b = pred_np.shape[0]
                    batch_x = batch_x[:keep_b]

                    if test_data.scale and self.args.inverse:
                        batch_x = test_data.inverse_transform(batch_x)
                        pred_vis = test_data.inverse_transform(pred_np)
                        true_vis = test_data.inverse_transform(true_np)
                    else:
                        pred_vis = pred_np
                        true_vis = true_np

                    gt = np.concatenate((batch_x[0, :, -1], true_vis[0, :, -1]), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1], pred_vis[0, :, -1]), axis=0)
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
