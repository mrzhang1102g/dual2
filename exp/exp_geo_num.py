"""
长期预测实验（GeoStyle，数值流 + element 元数据）
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
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from models import Model_Geo_Num, Model_Geo_Num_With_Meta

warnings.filterwarnings('ignore')


class Exp_Geo_Num(Exp_Basic):
    """
    GeoStyle 数值流实验类
    
    负责 GeoStyle 数据集的数值流模型训练和测试
    """

    def __init__(self, args):
        super(Exp_Geo_Num, self).__init__(args)
        # 模型字典
        self.model_dict = {
            'Model_Geo_Num': Model_Geo_Num,
            'Model_Geo_Num_With_Meta': Model_Geo_Num_With_Meta,
        }

    # ============================================================
    # Model
    # ============================================================
    def _build_model(self, args):
        model = self.model_dict[args.model](args).float()
        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # ============================================================
    # Data
    # ============================================================
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"[{flag.upper()}] {len(data_set)} samples (Geo_Meta)")
        return data_set, data_loader

    # ============================================================
    # Optimizer / Loss
    # ============================================================
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.loss == 'MAE':
            return nn.L1Loss()
        return nn.MSELoss()

    # ============================================================
    # Validation
    # ============================================================
    def vali(self, vali_loader, criterion, desc='Validation'):
        self.model.eval()
        losses = []

        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc=desc, unit='it')
            for _, batch in pbar:
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    element_ids,
                    group_ids,
                    norms
                ) = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                element_ids = element_ids.long().to(self.device)
                group_ids = group_ids.long().to(self.device)
                norms = norms.float().to(self.device)  # [B,3] -> [min,max,eps]

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_ids,
                    group_ids=group_ids,
                    caption_emb=None
                )

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]


                loss = criterion(outputs, target)
                losses.append(loss.item())

        self.model.train()
        return np.mean(losses)

    # ============================================================
    # Train
    # ============================================================
    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        # ===== 打印可训练参数（与 Fusion 一致）=====
        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
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
                    element_ids,
                    group_ids,
                    norms
                ) = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                element_ids = element_ids.long().to(self.device)
                group_ids = group_ids.long().to(self.device)
                norms = norms.float().to(self.device)  # [B,3] -> [min,max,eps]

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_ids,
                    group_ids=group_ids,
                    caption_emb=None
                )

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]


                loss = criterion(outputs, target)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_loader, criterion, desc='Validation')

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

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location='cpu')
        )
        return self.model

    # ============================================================
    # Test
    # ============================================================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')

        if test:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoint_dir, setting, 'checkpoint.pth'),
                    map_location='cpu'
                )
            )

        preds, trues = [], []

        result_folder = os.path.join(self.args.output_dir, setting, 'results/')
        os.makedirs(result_folder, exist_ok=True)

        vis_folder = os.path.join(self.args.output_dir, setting, 'visualizations/')
        if self.args.visualize:
            os.makedirs(vis_folder, exist_ok=True)

        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', unit='it')
            for i, batch in pbar: 
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    element_ids,
                    group_ids,
                    norms
                ) = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                element_ids = element_ids.long().to(self.device)
                group_ids = group_ids.long().to(self.device)
                norms = norms.float().to(self.device)  # [B,3] -> [min,max,eps]

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    dec_inp, batch_y_mark,
                    element_ids=element_ids,
                    group_ids=group_ids,
                    caption_emb=None
                )

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                # outputs, batch_y_cut: [B, pred_len, C]
                min_v = norms[:, 0].view(-1, 1, 1)
                max_v = norms[:, 1].view(-1, 1, 1)

                pred_denorm = outputs * (max_v - min_v) + min_v
                true_denorm = batch_y_cut * (max_v - min_v) + min_v

                preds.append(pred_denorm.detach().cpu().numpy())
                trues.append(true_denorm.detach().cpu().numpy())
                

                # =====================================================
                # PDF 可视化（统一在真实概率空间）
                # =====================================================
                if self.args.visualize and i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()

                    # ===== per-series denorm（GeoStyle 核心）=====
                    min_v_np = min_v.detach().cpu().numpy()
                    max_v_np = max_v.detach().cpu().numpy()
                    input_denorm = input_x * (max_v_np - min_v_np) + min_v_np

                    pred_np = pred_denorm.detach().cpu().numpy()
                    true_np = true_denorm.detach().cpu().numpy()

                    gt = np.concatenate(
                        (input_denorm[0, :, -1], true_np[0, :, -1]),
                        axis=0
                    )
                    pd = np.concatenate(
                        (input_denorm[0, :, -1], pred_np[0, :, -1]),
                        axis=0
                    )

                    T = input_denorm.shape[1]

                    from utils.tools import visual
                    visual(gt, pd, os.path.join(vis_folder, f'{i}.pdf'), history_len=T)

        self.log(f"Inference time: {time.time() - start_time:.2f}s")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if self.args.inverse:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)

        self.log('\n' + '=' * 50)
        self.log('Test Results (GeoStyle)')
        self.log('=' * 50)
        self.log(f'MAE:  {mae:.6f}')
        self.log(f'MSE:  {mse:.6f}')
        self.log(f'RMSE: {rmse:.6f}')
        self.log(f'MAPE: {mape * 100:.2f}%')
        self.log(f'MSPE: {mspe:.6f}')
        self.log(f'WAPE: {wape * 100:.2f}%')
        self.log('=' * 50 + '\n')

        np.save(result_folder + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe, wape]))
        np.save(result_folder + 'pred.npy', preds)
        np.save(result_folder + 'true.npy', trues)

        # 保存result.txt文件
        result_file_path = os.path.join(result_folder, 'result.txt')
        with open(result_file_path, 'w') as f:
            f.write(setting + "  \n")
            f.write(f'MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAPE: {mape*100:.2f}%, WAPE: {wape*100:.2f}%')
            f.write('\n\n')

        # 汇总 PDF - 暂时注释掉，统一输出结构
        # self._plot_predictions(
        #     preds, trues,
        #     result_folder, setting,
        #     mae=mae, mse=mse, rmse=rmse, mape=mape, wape=wape
        # )

        return mae, mse, rmse, mape, mspe, wape



    # ============================================================
    # 可视化
    # ============================================================
    def _plot_predictions( self, preds, trues, folder_path, setting, mae, mse, rmse, mape, wape, num_samples=10):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        reports_folder = os.path.join(self.args.output_dir, setting, 'reports/')
        os.makedirs(reports_folder, exist_ok=True)

        pdf_path = os.path.join(reports_folder, 'predictions.pdf')

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
                f'Model_Geo_Num_With_Meta Prediction Results\n{setting}',
                fontsize=14
            )

            ax = axes[0, 0]
            ax.scatter(trues.flatten()[::100],
                       preds.flatten()[::100],
                       alpha=0.3, s=1)
            ax.plot(
                [trues.min(), trues.max()],
                [trues.min(), trues.max()],
                'r--', lw=2
            )
            ax.set_title('Prediction vs Ground Truth')

            ax = axes[0, 1]
            errors = preds.flatten() - trues.flatten()
            ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_title(f'Error Distribution (Mean: {errors.mean():.4f})')

            ax = axes[1, 0]
            mae_per_step = np.mean(np.abs(preds - trues), axis=0).squeeze()
            ax.bar(range(len(mae_per_step)), mae_per_step)
            ax.set_title('MAE per Prediction Step')

            ax = axes[1, 1]
            ax.axis('off')
            table = ax.table(
                cellText=[
                    ['MAE', f'{mae:.6f}'],
                    ['MSE', f'{mse:.6f}'],
                    ['RMSE', f'{rmse:.6f}'],
                    ['MAPE', f'{mape * 100:.2f}%'],
                    ['WAPE', f'{wape * 100:.2f}%'],
                ],
                colLabels=['Metric', 'Value'],
                loc='center',
                cellLoc='center'
            )
            table.scale(1.2, 1.5)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            for idx in sample_indices:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(trues[idx].squeeze(), label='GT')
                ax.plot(preds[idx].squeeze(), label='Pred')
                ax.legend()
                ax.set_title(f'Sample {idx}')
                pdf.savefig(fig)
                plt.close()

        self.log(f'✅ Predictions PDF saved to: {pdf_path}')
