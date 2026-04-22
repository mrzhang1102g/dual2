"""Geo 数值流实验。"""

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
from models import Model_Geo_Num, Model_Geo_Num_With_Meta
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Geo_Num(Exp_Basic):
    """GeoStyle 数值流实验类。"""

    def __init__(self, args):
        super().__init__(args)
        self.model_dict = {
            "Model_Geo_Num": Model_Geo_Num,
            "Model_Geo_Num_With_Meta": Model_Geo_Num_With_Meta,
        }

    def _build_model(self, args):
        model = self.model_dict[args.model](args).float()
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples ({self.args.data})")
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.loss == "MAE":
            return nn.L1Loss()
        return nn.MSELoss()

    def _unpack_batch(self, batch):
        if len(batch) != 7:
            raise ValueError(
                "Geo numerical loader expects 7 items "
                "(x, y, x_mark, y_mark, element_ids, group_ids, norms), "
                f"got {len(batch)}"
            )
        return batch

    def _move_batch_to_device(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = self._unpack_batch(batch)
        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
            element_ids.long().to(self.device),
            group_ids.long().to(self.device),
            norms.float().to(self.device),
        )

    def _build_dec_inp(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).to(self.device)
        return dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids, group_ids):
        module = self.model.module if hasattr(self.model, "module") else self.model

        if isinstance(module, Model_Geo_Num_With_Meta):
            return self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                element_ids=element_ids,
                group_ids=group_ids,
                caption_emb=None,
            )

        return self.model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
        )

    def vali(self, vali_loader, criterion, desc="Validation"):
        self.model.eval()
        losses = []

        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc=desc, unit="it")
            for _, batch in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = self._move_batch_to_device(batch)
                dec_inp = self._build_dec_inp(batch_y)

                outputs = self._forward_model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    element_ids,
                    group_ids,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len :, f_dim:]
                target = batch_y[:, -self.args.pred_len :, f_dim:]
                losses.append(criterion(pred, target).item())

        self.model.train()
        return float(np.mean(losses))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        _, _ = self._get_data("test")

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

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

                batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = self._move_batch_to_device(batch)
                dec_inp = self._build_dec_inp(batch_y)

                outputs = self._forward_model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    element_ids,
                    group_ids,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len :, f_dim:]
                target = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = criterion(pred, target)
                train_losses.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = float(np.mean(train_losses))
            vali_loss = self.vali(vali_loader, criterion, desc="Validation")

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")

        if test:
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoint_dir, setting, "checkpoint.pth"), map_location="cpu")
            )

        preds, trues = [], []
        result_folder = os.path.join(self.args.output_dir, setting, "results")
        os.makedirs(result_folder, exist_ok=True)

        vis_folder = os.path.join(self.args.output_dir, setting, "visualizations")
        if self.args.visualize:
            os.makedirs(vis_folder, exist_ok=True)

        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", unit="it")
            for i, batch in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = self._move_batch_to_device(batch)
                dec_inp = self._build_dec_inp(batch_y)

                outputs = self._forward_model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    element_ids,
                    group_ids,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:]

                min_v = norms[:, 0].view(-1, 1, 1)
                max_v = norms[:, 1].view(-1, 1, 1)
                pred_denorm = outputs * (max_v - min_v) + min_v
                true_denorm = batch_y_cut * (max_v - min_v) + min_v

                preds.append(pred_denorm.detach().cpu().numpy())
                trues.append(true_denorm.detach().cpu().numpy())

                if self.args.visualize and i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    min_v_np = min_v.detach().cpu().numpy()
                    max_v_np = max_v.detach().cpu().numpy()
                    input_denorm = input_x * (max_v_np - min_v_np) + min_v_np

                    pred_np = pred_denorm.detach().cpu().numpy()
                    true_np = true_denorm.detach().cpu().numpy()

                    gt = np.concatenate((input_denorm[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_denorm[0, :, -1], pred_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(vis_folder, f"{i}.pdf"), history_len=input_denorm.shape[1])

        self.log(f"Inference time: {time.time() - start_time:.2f}s")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if self.args.inverse:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)

        self.log("\n" + "=" * 50)
        self.log("Test Results (GeoStyle)")
        self.log("=" * 50)
        self.log(f"MAE:  {mae:.6f}")
        self.log(f"MSE:  {mse:.6f}")
        self.log(f"RMSE: {rmse:.6f}")
        self.log(f"MAPE: {mape * 100:.2f}%")
        self.log(f"MSPE: {mspe:.6f}")
        self.log(f"WAPE: {wape * 100:.2f}%")
        self.log("=" * 50 + "\n")

        np.save(os.path.join(result_folder, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe, wape]))
        np.save(os.path.join(result_folder, "pred.npy"), preds)
        np.save(os.path.join(result_folder, "true.npy"), trues)

        result_file_path = os.path.join(result_folder, "result.txt")
        with open(result_file_path, "w", encoding="utf-8") as file:
            file.write(setting + "  \n")
            file.write(
                f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, "
                f"MAPE: {mape * 100:.2f}%, WAPE: {wape * 100:.2f}%"
            )
            file.write("\n\n")

        return mae, mse, rmse, mape, mspe, wape
