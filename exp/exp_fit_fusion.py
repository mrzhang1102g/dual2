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
from models.model_fit_fusion import Model_Fit_Fusion as FusionModel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')


class Exp_Fit_Fusion(Exp_Basic):
    """
    DualSG-Fit-Fusion 实验类

    - 数值流 + 文本流（caption embedding）
    - 所有 batch 结构固定为 FIT_Fusion 返回格式
    """

    def __init__(self, args):
        super().__init__(args)

    # ============================================================
    # 构建 Fusion 模型
    # ============================================================
    def _build_model(self, args):


        num_ckpt = getattr(args, "num_model_path", None)
        self.log(f"num_ckpt: {num_ckpt}")

        model = FusionModel(
            args,
            numerical_ckpt_path=num_ckpt
        ).float()

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model

    # ============================================================
    # 数据加载
    # ============================================================
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.log(f"{flag} {len(data_set)} samples (FIT Fusion)")
        return data_set, data_loader

    # ============================================================
    # Optimizer
    # ============================================================
    def _select_optimizer(self):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate
        )

    # ============================================================
    # Criterion
    # ============================================================
    def _select_criterion(self):
        if self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAE':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.args.loss}")

    # ============================================================
    # 1. 训练
    # ============================================================
    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoint_dir, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # ===== 初始化模型参数 =====
        # 先运行一次前向传播来初始化所有参数
        for _, batch in enumerate(train_loader):
            (batch_x, batch_y, batch_x_mark, batch_y_mark, city, gender, age, element, caption_emb) = batch
            
            # 维度修正
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1)
            if batch_y.dim() == 2:
                batch_y = batch_y.unsqueeze(-1)
            
            # 移动到正确的设备
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            city = city.long().to(self.device)
            gender = gender.long().to(self.device)
            age = age.long().to(self.device)
            element = element.long().to(self.device)
            caption_emb = caption_emb.float().to(self.device)
            
            # 初始化参数
            with torch.no_grad():
                self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, city, gender, age, element, caption_emb)
            break

        # ===== 打印可训练参数 =====
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
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                ) = batch

                model_optim.zero_grad()

                # ---- 维度修正 ----
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                city_id = city_id.long().to(self.device)
                gender_id = gender_id.long().to(self.device)
                age_id = age_id.long().to(self.device)
                element_id = element_id.long().to(self.device)
                caption_emb = caption_emb.float().to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    batch_y, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                )

                target = batch_y[:, -self.args.pred_len:, -1:]
                # loss
                loss_main  = criterion(outputs, target)

                # ===== aux_loss（纠偏正则，可选）=====
                m = self.model.module if hasattr(self.model, "module") else self.model
                aux_loss = getattr(m, "aux_loss", None)

                loss = loss_main
                if aux_loss is not None:
                    loss = loss + aux_loss

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.mean(train_loss)
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

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location='cpu')
        )
        return self.model

    # ============================================================
    # 2. 验证
    # ============================================================
    def vali(self, vali_loader, criterion):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in vali_loader:
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                ) = batch

                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                city_id = city_id.long().to(self.device)
                gender_id = gender_id.long().to(self.device)
                age_id = age_id.long().to(self.device)
                element_id = element_id.long().to(self.device)
                caption_emb = caption_emb.float().to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    batch_y, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                )

                target = batch_y[:, -self.args.pred_len:, -1:]
                loss = criterion(outputs, target)
                losses.append(loss.item())

        self.model.train()
        return np.mean(losses)

    # ============================================================
    # 3. 测试
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

        self.model.eval()
        preds, trues = [], []

        # ========= 新增：只跑前N个test样本（默认-1/None表示全量）=========
        max_n = getattr(self.args, "max_test_samples", -1)
        if max_n is None:
            max_n = -1
        max_n = int(max_n)

        seen = 0  # 已收集的样本数

        # 可视化目录
        vis_folder = os.path.join(self.args.output_dir, setting, 'visualizations/')
        # 无论是否可视化，都创建目录（因为后面代码可能会用到）
        os.makedirs(vis_folder, exist_ok=True)

        self.log(f'[TEST] Batch-level PDFs will be saved to: {vis_folder}')
        if max_n > 0:
            self.log(f'[TEST] Will evaluate only first {max_n} test samples.')

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (
                    batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                ) = batch

                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                city_id = city_id.long().to(self.device)
                gender_id = gender_id.long().to(self.device)
                age_id = age_id.long().to(self.device)
                element_id = element_id.long().to(self.device)
                caption_emb = caption_emb.float().to(self.device)

                outputs = self.model(
                    batch_x, batch_x_mark,
                    batch_y, batch_y_mark,
                    city_id, gender_id, age_id, element_id,
                    caption_emb
                )

                # ========= 新增：截断到前N个样本 =========
                pred_np = outputs.detach().cpu().numpy()
                true_np = batch_y[:, -self.args.pred_len:, -1:].detach().cpu().numpy()

                if max_n > 0:
                    remain = max_n - seen
                    if remain <= 0:
                        break
                    if pred_np.shape[0] > remain:
                        pred_np = pred_np[:remain]
                        true_np = true_np[:remain]

                # append（保持原逻辑：收集到列表，最后concat）
                preds.append(pred_np)
                trues.append(true_np)
                seen += pred_np.shape[0]

                # =====================================================
                # batch-level PDF（保持原逻辑，只加一个小保护：当前batch确实有样本）
                # =====================================================
                if i % 20 == 0 and pred_np.shape[0] > 0:
                    input_x = batch_x.detach().cpu().numpy()
                    pred_vis = outputs.detach().cpu().numpy()
                    true_vis = batch_y[:, -self.args.pred_len:, -1:].detach().cpu().numpy()

                    # 若开启max_n且发生截断，为了可视化一致，也截断可视化数组
                    if max_n > 0:
                        # 当前batch实际保留的数量 = pred_np.shape[0]
                        keep_b = pred_np.shape[0]
                        input_x = input_x[:keep_b]
                        pred_vis = pred_vis[:keep_b]
                        true_vis = true_vis[:keep_b]

                    if self.args.inverse:
                        shape = input_x.shape
                        input_x = test_data.inverse_transform(
                            input_x.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                        shape = pred_vis.shape
                        pred_vis = test_data.inverse_transform(
                            pred_vis.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                        shape = true_vis.shape
                        true_vis = test_data.inverse_transform(
                            true_vis.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                    # 只画 batch 中第0个样本
                    gt = np.concatenate(
                        (input_x[0, :, -1], true_vis[0, :, -1]),
                        axis=0
                    )
                    pd = np.concatenate(
                        (input_x[0, :, -1], pred_vis[0, :, -1]),
                        axis=0
                    )

                    from utils.tools import visual
                    pdf_path = os.path.join(vis_folder, f'{i}.pdf')
                    visual(gt, pd, pdf_path)
                    self.log(f'[TEST][PDF] Saved batch-level PDF: {pdf_path}')

                # 满足N后提前结束
                if max_n > 0 and seen >= max_n:
                    break

        preds = np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([])
        trues = np.concatenate(trues, axis=0) if len(trues) > 0 else np.array([])

        if self.args.inverse and preds.size > 0:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)

        self.log('\n' + '=' * 50)
        self.log('Test Results (FIT)')
        self.log('=' * 50)
        self.log(f'MAE:  {mae:.6f}')
        self.log(f'MSE:  {mse:.6f}')
        self.log(f'RMSE: {rmse:.6f}')
        self.log(f'MAPE: {mape * 100:.2f}%')
        self.log(f'MSPE: {mspe:.6f}')
        self.log(f'WAPE: {wape * 100:.2f}%')
        self.log('=' * 50)

        # =====================================================
        # 汇总 PDF（保持原逻辑：用收集到的preds/trues做总图）
        # =====================================================
        if preds.size > 0:
            self._plot_predictions(
                preds, trues,
                vis_folder, setting,
                mae=mae, mse=mse, rmse=rmse, mape=mape, wape=wape
            )
            self.log(f'[TEST][PDF] Saved summary PDF: {vis_folder}predictions.pdf')

        return mae, mse, rmse, mape, mspe, wape

    # ============================================================
    # Visualization
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
                f'Model_Fit_Fusion Prediction Results\n{setting}',
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