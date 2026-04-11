"""
FIT 数值流实验类

负责 FIT 数据集的数值流模型训练和测试
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models import Model_Fit_Num, Model_Fit_Num_With_Meta, Model_Geo_Num, Model_Geo_Num_With_Meta, Model_Fit_Fusion
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from tqdm import tqdm
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')


class Exp_Fit_Num(Exp_Basic):
    """
    FIT 数值流实验类
    
    负责 FIT 数据集的数值流模型训练和测试
    """

    def __init__(self, args):
        """
        初始化 FIT 数值流实验类
        
        参数:
            args: 命令行参数
        """
        super(Exp_Fit_Num, self).__init__(args)

    def _build_model(self, args):
        """
        构建模型
        
        参数:
            args: 命令行参数
            
        返回:
            model: 构建好的模型
        """
        if args.model == "Model_Fit_Fusion":
            model = Model_Fit_Fusion(
                args,
                numerical_ckpt_path=args.num_model_path
            ).float()
        else:
            model = self.model_dict[args.model](args).float()

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model

    def _get_data(self, flag):
        """
        获取数据
        
        参数:
            flag: 数据类型，可选值为 'train', 'val', 'test'
            
        返回:
            data_set: 数据集
            data_loader: 数据加载器
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择优化器
        
        返回:
            model_optim: 优化器
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数
        
        返回:
            criterion: 损失函数
        """
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion, desc='Validation'):
        """
        验证模型
        
        参数:
            vali_data: 验证数据集
            vali_loader: 验证数据加载器
            criterion: 损失函数
            desc: 进度条描述
            
        返回:
            total_loss: 平均损失
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc=desc, unit='it')
            for i, batch in pbar:
                if self.args.model == "Model_Fit_Fusion":
                    # 融合模型需要额外的annotations
                    if len(batch) == 5:
                        batch_x, batch_y, batch_x_mark, batch_y_mark, annotations = batch
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    # GeoStyle带元数据模型：7个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = batch
                    annotations = None
                elif self.args.model == "Model_Fit_Num_With_Meta":
                    # FIT带元数据模型：8个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, city_ids, gender_ids, age_ids, element_ids = batch
                    annotations = None
                else:
                    # 纯数值模型可能返回更多值（如元数据）
                    if len(batch) >= 4:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                        annotations = None
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 把元数据也移到device上
                if self.args.model == "Model_Fit_Num_With_Meta":
                    city_ids = city_ids.to(self.device)
                    gender_ids = gender_ids.to(self.device)
                    age_ids = age_ids.to(self.device)
                    element_ids = element_ids.to(self.device)
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    element_ids = element_ids.to(self.device)
                    group_ids = group_ids.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == "Model_Fit_Fusion":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                        elif self.args.model == "Model_Geo_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                        elif self.args.model == "Model_Fit_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == "Model_Fit_Fusion":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                    elif self.args.model == "Model_Geo_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                    elif self.args.model == "Model_Fit_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        训练模型
        
        参数:
            setting: 实验设置
            
        返回:
            model: 训练好的模型
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoint_dir, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ===== 打印可学习参数 =====
        self.print_trainable_parameters()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}', unit='it')
            for i, batch in pbar:
                if self.args.model == "Model_Fit_Fusion":
                    # 融合模型需要额外的annotations
                    if len(batch) == 5:
                        batch_x, batch_y, batch_x_mark, batch_y_mark, annotations = batch
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    # GeoStyle带元数据模型：7个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = batch
                    annotations = None
                elif self.args.model == "Model_Fit_Num_With_Meta":
                    # FIT带元数据模型：8个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, city_ids, gender_ids, age_ids, element_ids = batch
                    annotations = None
                else:
                    # 纯数值模型可能返回更多值（如元数据）
                    if len(batch) >= 4:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                        annotations = None
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 把元数据也移到device上
                if self.args.model == "Model_Fit_Num_With_Meta":
                    city_ids = city_ids.to(self.device)
                    gender_ids = gender_ids.to(self.device)
                    age_ids = age_ids.to(self.device)
                    element_ids = element_ids.to(self.device)
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    element_ids = element_ids.to(self.device)
                    group_ids = group_ids.to(self.device)
                
                # 把元数据也移到device上
                if self.args.model == "Model_Fit_Num_With_Meta":
                    city_ids = city_ids.to(self.device)
                    gender_ids = gender_ids.to(self.device)
                    age_ids = age_ids.to(self.device)
                    element_ids = element_ids.to(self.device)
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    element_ids = element_ids.to(self.device)
                    group_ids = group_ids.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == "Model_Fit_Fusion":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                        elif self.args.model == "Model_Geo_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                        elif self.args.model == "Model_Fit_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == "Model_Fit_Fusion":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                    elif self.args.model == "Model_Geo_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                    elif self.args.model == "Model_Fit_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.log("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.log('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
            self.log("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, desc='Validation')

            self.log("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.log("Early stopping")
                break
            if self.args.adjust:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))

        return self.model

    def test(self, setting, test=0):
        """
        测试模型
        
        参数:
            setting: 实验设置
            test: 是否加载已训练的模型
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoint_dir, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        preds = []
        trues = []
        
        # 使用配置的输出目录
        folder_path = os.path.join(self.args.output_dir, setting, 'visualizations/')
        if self.args.visualize and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        start_time_test = time.time()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', unit='it')
            for i, batch in pbar:
                if self.args.model == "Model_Fit_Fusion":
                    # 融合模型需要额外的annotations
                    if len(batch) == 5:
                        batch_x, batch_y, batch_x_mark, batch_y_mark, annotations = batch
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    # GeoStyle带元数据模型：7个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, element_ids, group_ids, norms = batch
                    annotations = None
                elif self.args.model == "Model_Fit_Num_With_Meta":
                    # FIT带元数据模型：8个返回值
                    batch_x, batch_y, batch_x_mark, batch_y_mark, city_ids, gender_ids, age_ids, element_ids = batch
                    annotations = None
                else:
                    # 纯数值模型可能返回更多值（如元数据）
                    if len(batch) >= 4:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                        annotations = None
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        annotations = None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 把元数据也移到device上
                if self.args.model == "Model_Fit_Num_With_Meta":
                    city_ids = city_ids.to(self.device)
                    gender_ids = gender_ids.to(self.device)
                    age_ids = age_ids.to(self.device)
                    element_ids = element_ids.to(self.device)
                elif self.args.model == "Model_Geo_Num_With_Meta":
                    element_ids = element_ids.to(self.device)
                    group_ids = group_ids.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.model == "Model_Fit_Fusion":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                        elif self.args.model == "Model_Geo_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                        elif self.args.model == "Model_Fit_Num_With_Meta":
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.model == "Model_Fit_Fusion":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, annotations)
                    elif self.args.model == "Model_Geo_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, element_ids=element_ids, group_ids=group_ids)
                    elif self.args.model == "Model_Fit_Num_With_Meta":
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, city_ids=city_ids, gender_ids=gender_ids, age_ids=age_ids, element_ids=element_ids)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # 仅在开启可视化时生成图表
                if self.args.visualize and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        end_time_test = time.time()
        self.log(f"Inference time: {end_time_test - start_time_test:.2f} seconds")
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.log(f'test shape: {preds.shape} {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        self.log(f'test shape: {preds.shape} {trues.shape}')

        # result save
        results_folder = os.path.join(self.args.output_dir, setting, 'results/')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    self.log("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)
        
        # 打印所有指标
        self.log('\n' + '='*50)
        self.log('Test Results:')
        self.log('='*50)
        self.log(f'MAE:  {mae:.6f}')
        self.log(f'MSE:  {mse:.6f}')
        self.log(f'RMSE: {rmse:.6f}')
        self.log(f'MAPE: {mape*100:.2f}%')
        self.log(f'MSPE: {mspe:.6f}')
        self.log(f'WAPE: {wape*100:.2f}%')
        if self.args.use_dtw:
            self.log(f'DTW:  {dtw:.6f}')
        self.log('='*50 + '\n')
        
        # 将结果写入到对应的outputs文件夹
        result_file_path = os.path.join(results_folder, 'result.txt')
        f = open(result_file_path, 'w')
        f.write(setting + "  \n")
        f.write(f'MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAPE: {mape*100:.2f}%, WAPE: {wape*100:.2f}%')
        f.write('\n\n')
        f.close()

        np.save(results_folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, wape]))
        np.save(results_folder + 'pred.npy', preds)
        np.save(results_folder + 'true.npy', trues)
        
        # 仅在开启可视化时生成PDF
        if self.args.visualize:
            self._plot_predictions(preds, trues, setting)

        return
    
    def _plot_predictions(self, preds, trues, setting, num_samples=10):
        """生成预测结果的可视化 PDF"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        # 确保输出文件夹存在
        pdf_folder = os.path.join(self.args.output_dir, setting, 'reports/')
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
        
        pdf_path = os.path.join(pdf_folder, 'predictions.pdf')
        
        # 随机选择样本进行可视化
        np.random.seed(self.args.seed)
        total_samples = preds.shape[0]
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        with PdfPages(pdf_path) as pdf:
            # 第一页：整体统计
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Model_Fit_Num Prediction Results\n{setting}', fontsize=14)
            
            # 1. 预测值 vs 真实值散点图
            ax = axes[0, 0]
            pred_flat = preds.flatten()
            true_flat = trues.flatten()
            ax.scatter(true_flat[::100], pred_flat[::100], alpha=0.3, s=1)
            ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Prediction vs Ground Truth')
            
            # 2. 误差分布直方图
            ax = axes[0, 1]
            errors = pred_flat - true_flat
            ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Error Distribution (Mean: {errors.mean():.4f})')
            
            # 3. 各时间步的 MAE
            ax = axes[1, 0]
            mae_per_step = np.mean(np.abs(preds - trues), axis=0).squeeze()
            ax.bar(range(len(mae_per_step)), mae_per_step, color='steelblue')
            ax.set_xlabel('Prediction Step (Week)')
            ax.set_ylabel('MAE')
            ax.set_title('MAE per Prediction Step')
            
            # 4. 指标汇总表格
            ax = axes[1, 1]
            ax.axis('off')
            mae, mse, rmse, mape, mspe, wape = metric(preds, trues)
            table_data = [
                ['MAE', f'{mae:.6f}'],
                ['MSE', f'{mse:.6f}'],
                ['RMSE', f'{rmse:.6f}'],
                ['MAPE', f'{mape*100:.2f}%'],
                ['WAPE', f'{wape*100:.2f}%']
            ]
            table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax.set_title('Metrics Summary')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # 后续页面：单个样本预测曲线
            for idx in sample_indices:
                fig, ax = plt.subplots(figsize=(12, 4))
                
                pred_seq = preds[idx].squeeze()
                true_seq = trues[idx].squeeze()
                
                weeks = range(1, len(pred_seq) + 1)
                ax.plot(weeks, true_seq, 'b-', label='Ground Truth', linewidth=2, marker='o', markersize=4)
                ax.plot(weeks, pred_seq, 'r--', label='Prediction', linewidth=2, marker='s', markersize=4)
                
                ax.fill_between(weeks, true_seq, pred_seq, alpha=0.3, color='gray')
                
                sample_mae = np.mean(np.abs(pred_seq - true_seq))
                ax.set_xlabel('Week')
                ax.set_ylabel('Trend Value')
                ax.set_title(f'Sample {idx} - 24-Week Prediction (MAE: {sample_mae:.4f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        self.log(f'✅ Predictions PDF saved to: {pdf_path}')
