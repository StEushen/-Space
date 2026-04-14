from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _is_ppn_model(self):
        return str(getattr(self.args, 'model', '')).lower() in {'ppn', 'tauonly'}

    def _is_tauonly_model(self):
        return str(getattr(self.args, 'model', '')).lower() == 'tauonly'

    def _apply_tauonly_anchor_schedule(self, epoch_idx):
        if not self._is_tauonly_model():
            return

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model_ref, 'anchor_active_scale'):
            return

        delay_epochs = max(0, int(getattr(self.args, 'tauonly_anchor_delay_epochs', 0)))
        base_scale = float(getattr(self.args, 'tauonly_anchor_scale', 1.0))
        model_ref.anchor_active_scale = 0.0 if epoch_idx < delay_epochs else base_scale

    def _compute_ppn_tau_loss(self, batch_x):
        if not self._is_ppn_model() or int(getattr(self.args, 'ppn_use_tau_loss', 1)) == 0:
            return torch.zeros((), device=batch_x.device, dtype=batch_x.dtype)

        if int(getattr(self.args, 'ppn_use_tau_space_predictor', 1)) == 1 and int(getattr(self.args, 'ppn_disable_accel_tau_loss_in_axiom_mode', 1)) == 1:
            return torch.zeros((), device=batch_x.device, dtype=batch_x.dtype)

        if batch_x.shape[1] < 4:
            return torch.zeros((), device=batch_x.device, dtype=batch_x.dtype)

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        tau_module = getattr(model_ref, 'tau_module', None)
        if tau_module is None:
            return torch.zeros((), device=batch_x.device, dtype=batch_x.dtype)

        x_t = batch_x[:, -1, :]
        x_t1 = batch_x[:, -2, :]
        x_t2 = batch_x[:, -3, :]
        x_t3 = batch_x[:, -4, :]

        # Use model-specific tau computation.
        if hasattr(model_ref, '_compute_simple_tau'):
            tau_t = model_ref._compute_simple_tau(x_t, x_t1, x_t2)
        elif hasattr(model_ref, '_compute_tau_field'):
            tau_t = model_ref._compute_tau_field(x_t, x_t1, x_t2)
        else:
            v_t = x_t - x_t1
            v_t1 = x_t1 - x_t2
            v_t_norm = torch.norm(v_t, dim=-1, keepdim=True).clamp_min(1e-6)
            v_t1_norm = torch.norm(v_t1, dim=-1, keepdim=True).clamp_min(1e-6)
            tau_t = v_t_norm / (v_t1_norm.clamp_min(1e-6) + 1e-6)
        a_t = x_t - 2 * x_t1 + x_t2
        a_t1 = x_t1 - 2 * x_t2 + x_t3
        delta_a = torch.norm(a_t - a_t1, dim=-1, keepdim=True)

        tau_norm = tau_t / (tau_t.mean().detach() + 1e-6)
        accel_norm = delta_a / (delta_a.mean().detach() + 1e-6)
        return ((tau_norm - accel_norm) ** 2).mean()

    def _compute_ppn_axiom_losses(self, ref_tensor):
        if not self._is_ppn_model():
            zero = torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)
            return zero, zero, zero

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        tau_recon = getattr(model_ref, 'last_tau_recon_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))
        tau_flat = getattr(model_ref, 'last_tau_flatness_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))
        tau_mono = getattr(model_ref, 'last_tau_monotonic_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))
        return tau_recon, tau_flat, tau_mono

    def _compute_ppn_traj_smooth_loss(self, ref_tensor):
        if not self._is_ppn_model():
            return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        return getattr(model_ref, 'last_tau_traj_smooth_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))

    def _compute_ppn_proj_cycle_loss(self, ref_tensor):
        if not self._is_ppn_model():
            return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        return getattr(model_ref, 'last_tau_proj_cycle_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))

    def _compute_ppn_contrast_loss(self, ref_tensor):
        if not self._is_ppn_model():
            return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        return getattr(model_ref, 'last_tau_contrast_loss', torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype))

    def _get_tau_aux_scale(self, epoch_idx):
        if int(getattr(self.args, 'ppn_use_tau_phase_schedule', 1)) != 1:
            return 1.0

        total_epochs = max(1, int(getattr(self.args, 'train_epochs', 1)))
        progress = float(epoch_idx + 1) / float(total_epochs)
        p1 = float(getattr(self.args, 'ppn_tau_phase1_ratio', 0.3))
        p2 = float(getattr(self.args, 'ppn_tau_phase2_ratio', 0.5))

        p1_end = max(0.0, min(1.0, p1))
        p2_end = max(p1_end, min(1.0, p1_end + p2))

        if progress <= p1_end:
            return float(getattr(self.args, 'ppn_tau_phase1_aux_scale', 0.3))
        if progress <= p2_end:
            return float(getattr(self.args, 'ppn_tau_phase2_aux_scale', 1.0))
        return float(getattr(self.args, 'ppn_tau_phase3_aux_scale', 1.2))

    def _use_partition_correction(self):
        return self._is_ppn_model() and int(getattr(self.args, 'ppn_use_partition_correction', 0)) == 1

    def _estimate_global_step_mse(self, data_loader):
        pred_len = int(self.args.pred_len)
        f_dim = -1 if self.args.features == 'MS' else 0
        accum = torch.zeros(pred_len, device=self.device)
        count = 0

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                step_mse = ((outputs - target) ** 2).mean(dim=(0, 2))
                accum += step_mse
                count += 1

        self.model.train()
        if count == 0:
            return torch.zeros(pred_len, device=self.device)
        return accum / float(count)

    def _compute_partition_correction_loss(self, outputs, target, global_step_mse):
        if global_step_mse is None:
            return torch.zeros((), device=outputs.device, dtype=outputs.dtype)
        margin = float(getattr(self.args, 'ppn_corr_margin', 0.0))
        local_step_mse = ((outputs - target) ** 2).mean(dim=(0, 2))
        correction = torch.relu(local_step_mse - global_step_mse + margin)
        return correction.mean()

    def _build_partition_loaders(self, train_data):
        n_groups = max(1, int(getattr(self.args, 'ppn_n_groups', 1)))
        m_cycles = max(1, int(getattr(self.args, 'ppn_m_cycles', 1)))
        total = len(train_data)
        if total == 0:
            return []

        indices = np.arange(total)
        chunks = np.array_split(indices, n_groups)
        chunks = [c for c in chunks if len(c) > 0]

        loaders = []
        for _ in range(m_cycles):
            for chunk in chunks:
                subset = Subset(train_data, chunk.tolist())
                loader = DataLoader(
                    subset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    drop_last=False,
                )
                loaders.append(loader)
        return loaders

    def _apply_ppn_residual_scale_schedule(self, epoch_idx):
        if not self._is_ppn_model():
            return

        if int(getattr(self.args, 'ppn_use_horizon_residual', 0)) != 1:
            return

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model_ref, 'horizon_residual_scale'):
            return

        base_scale = float(getattr(self.args, 'ppn_horizon_residual_scale', 0.1))
        use_warmup = int(getattr(self.args, 'ppn_use_horizon_residual_warmup', 0)) == 1
        warmup_epochs = max(1, int(getattr(self.args, 'ppn_horizon_residual_warmup_epochs', 1)))

        if use_warmup:
            progress = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
            current_scale = base_scale * progress
        else:
            current_scale = base_scale

        model_ref.horizon_residual_scale = current_scale
 

    def vali(self, vali_data, vali_loader, criterion, epoch_idx=0):
        total_loss = []
        tau_aux_scale = self._get_tau_aux_scale(epoch_idx)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                mse_loss = criterion(pred, true)
                tau_loss = self._compute_ppn_tau_loss(batch_x)
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                tau_smooth_loss = getattr(model_ref, 'last_tau_smooth_loss', torch.zeros((), device=pred.device, dtype=pred.dtype))
                tau_traj_smooth_loss = self._compute_ppn_traj_smooth_loss(pred)
                tau_proj_cycle_loss = self._compute_ppn_proj_cycle_loss(pred)
                tau_contrast_loss = self._compute_ppn_contrast_loss(pred)
                tau_recon_loss, tau_flat_loss, tau_mono_loss = self._compute_ppn_axiom_losses(pred)
                loss = (
                    mse_loss
                    + float(getattr(self.args, 'ppn_lambda_tau', 0.2)) * tau_loss
                    + float(getattr(self.args, 'ppn_lambda_tau_smooth', 0.0)) * tau_smooth_loss
                    + float(getattr(self.args, 'ppn_lambda_tau_traj_smooth', 0.0)) * tau_traj_smooth_loss
                    + float(getattr(self.args, 'ppn_lambda_proj_cycle', 0.0)) * tau_proj_cycle_loss
                    + float(getattr(self.args, 'ppn_lambda_tau_contrast', 0.0)) * tau_contrast_loss
                    + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_recon', 0.05)) * tau_recon_loss
                    + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_flat', 0.01)) * tau_flat_loss
                    + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_mono', 0.01)) * tau_mono_loss
                )

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        subset_ratio = float(getattr(self.args, 'train_subset_ratio', 1.0))
        effective_train_data = train_data
        if 0 < subset_ratio < 1.0:
            total = len(train_data)
            subset_size = max(1, int(total * subset_ratio))
            rng = np.random.RandomState(int(getattr(self.args, 'subset_seed', 42)))
            subset_idx = rng.choice(total, size=subset_size, replace=False)
            subset_data = Subset(train_data, subset_idx.tolist())
            effective_train_data = subset_data
            train_loader = DataLoader(
                subset_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
            print(f"Few-shot mode: using {subset_size}/{total} train samples ({subset_ratio:.2%})")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        partition_mode = self._use_partition_correction()
        if partition_mode:
            partition_loaders = self._build_partition_loaders(effective_train_data)
            if partition_loaders:
                train_steps = sum(len(loader) for loader in partition_loaders)
            print(
                f"Partition correction mode ON: n={int(getattr(self.args, 'ppn_n_groups', 1))}, "
                f"m={int(getattr(self.args, 'ppn_m_cycles', 1))}, steps={train_steps}"
            )
        else:
            partition_loaders = [train_loader]

        for epoch in range(self.args.train_epochs):
            self._apply_ppn_residual_scale_schedule(epoch)
            self._apply_tauonly_anchor_schedule(epoch)
            tau_aux_scale = self._get_tau_aux_scale(epoch)
            iter_count = 0
            train_loss = []
            tau_losses = []
            tau_smooth_losses = []
            tau_traj_smooth_losses = []
            tau_proj_cycle_losses = []
            tau_contrast_losses = []
            tau_recon_losses = []
            tau_flat_losses = []
            tau_mono_losses = []
            corr_losses = []

            if partition_mode:
                global_loader = DataLoader(
                    effective_train_data,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    drop_last=False,
                )
                global_step_mse = self._estimate_global_step_mse(global_loader)
            else:
                global_step_mse = None

            self.model.train()
            epoch_time = time.time()
            global_i = 0
            for current_loader in partition_loaders:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(current_loader):
                    iter_count += 1
                    global_i += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            mse_loss = criterion(outputs, batch_y)
                            tau_loss = self._compute_ppn_tau_loss(batch_x)
                            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                            tau_smooth_loss = getattr(model_ref, 'last_tau_smooth_loss', torch.zeros((), device=outputs.device, dtype=outputs.dtype))
                            tau_traj_smooth_loss = self._compute_ppn_traj_smooth_loss(outputs)
                            tau_proj_cycle_loss = self._compute_ppn_proj_cycle_loss(outputs)
                            tau_contrast_loss = self._compute_ppn_contrast_loss(outputs)
                            tau_recon_loss, tau_flat_loss, tau_mono_loss = self._compute_ppn_axiom_losses(outputs)
                            corr_loss = self._compute_partition_correction_loss(outputs, batch_y, global_step_mse)
                            loss = (
                                mse_loss
                                + float(getattr(self.args, 'ppn_lambda_tau', 0.2)) * tau_loss
                                + float(getattr(self.args, 'ppn_lambda_tau_smooth', 0.0)) * tau_smooth_loss
                                + float(getattr(self.args, 'ppn_lambda_tau_traj_smooth', 0.0)) * tau_traj_smooth_loss
                                + float(getattr(self.args, 'ppn_lambda_proj_cycle', 0.0)) * tau_proj_cycle_loss
                                + float(getattr(self.args, 'ppn_lambda_tau_contrast', 0.0)) * tau_contrast_loss
                                + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_recon', 0.05)) * tau_recon_loss
                                + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_flat', 0.01)) * tau_flat_loss
                                + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_mono', 0.01)) * tau_mono_loss
                                + float(getattr(self.args, 'ppn_lambda_corr', 0.1)) * corr_loss
                            )
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        mse_loss = criterion(outputs, batch_y)
                        tau_loss = self._compute_ppn_tau_loss(batch_x)
                        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                        tau_smooth_loss = getattr(model_ref, 'last_tau_smooth_loss', torch.zeros((), device=outputs.device, dtype=outputs.dtype))
                        tau_traj_smooth_loss = self._compute_ppn_traj_smooth_loss(outputs)
                        tau_proj_cycle_loss = self._compute_ppn_proj_cycle_loss(outputs)
                        tau_contrast_loss = self._compute_ppn_contrast_loss(outputs)
                        tau_recon_loss, tau_flat_loss, tau_mono_loss = self._compute_ppn_axiom_losses(outputs)
                        corr_loss = self._compute_partition_correction_loss(outputs, batch_y, global_step_mse)
                        loss = (
                            mse_loss
                            + float(getattr(self.args, 'ppn_lambda_tau', 0.2)) * tau_loss
                            + float(getattr(self.args, 'ppn_lambda_tau_smooth', 0.0)) * tau_smooth_loss
                            + float(getattr(self.args, 'ppn_lambda_tau_traj_smooth', 0.0)) * tau_traj_smooth_loss
                            + float(getattr(self.args, 'ppn_lambda_proj_cycle', 0.0)) * tau_proj_cycle_loss
                            + float(getattr(self.args, 'ppn_lambda_tau_contrast', 0.0)) * tau_contrast_loss
                            + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_recon', 0.05)) * tau_recon_loss
                            + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_flat', 0.01)) * tau_flat_loss
                            + tau_aux_scale * float(getattr(self.args, 'ppn_lambda_tau_mono', 0.01)) * tau_mono_loss
                            + float(getattr(self.args, 'ppn_lambda_corr', 0.1)) * corr_loss
                        )

                    train_loss.append(loss.item())
                    tau_losses.append(tau_loss.item())
                    tau_smooth_losses.append(tau_smooth_loss.item())
                    tau_traj_smooth_losses.append(tau_traj_smooth_loss.item())
                    tau_proj_cycle_losses.append(tau_proj_cycle_loss.item())
                    tau_contrast_losses.append(tau_contrast_loss.item())
                    tau_recon_losses.append(tau_recon_loss.item())
                    tau_flat_losses.append(tau_flat_loss.item())
                    tau_mono_losses.append(tau_mono_loss.item())
                    corr_losses.append(corr_loss.item())

                    if global_i % 100 == 0:
                        if self._is_ppn_model():
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f} tau: {3:.7f} corr: {4:.7f}".format(
                                global_i, epoch + 1, loss.item(), tau_loss.item(), corr_loss.item()
                            ))
                        else:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(global_i, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - global_i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)
            test_loss = self.vali(test_data, test_loader, criterion, epoch)

            if self._is_ppn_model() and tau_losses:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Tau Loss: {3:.7f} Corr Loss: {4:.7f} Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    float(np.average(tau_losses)),
                    float(np.average(corr_losses)) if corr_losses else 0.0,
                    vali_loss,
                    test_loss,
                ))
                if tau_smooth_losses:
                    print("Tau Smooth Loss: {0:.7f}".format(float(np.average(tau_smooth_losses))))
                if tau_traj_smooth_losses:
                    print("Tau Traj Smooth: {0:.7f}".format(float(np.average(tau_traj_smooth_losses))))
                if tau_proj_cycle_losses:
                    print("Tau Proj Cycle: {0:.7f}".format(float(np.average(tau_proj_cycle_losses))))
                if tau_contrast_losses:
                    print("Tau Contrast: {0:.7f}".format(float(np.average(tau_contrast_losses))))
                if tau_recon_losses:
                    print("Tau Recon Loss:  {0:.7f}".format(float(np.average(tau_recon_losses))))
                if tau_flat_losses:
                    print("Tau Flat Loss:   {0:.7f}".format(float(np.average(tau_flat_losses))))
                if tau_mono_losses:
                    print("Tau Mono Loss:   {0:.7f}".format(float(np.average(tau_mono_losses))))
                print("Tau Aux Scale:   {0:.3f}".format(float(tau_aux_scale)))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
