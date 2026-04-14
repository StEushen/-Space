import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--ppn_use_tau_loss', type=int, default=1,
                        help='PPN only: enable intrinsic-time regularization loss (1/0)')
    parser.add_argument('--ppn_lambda_tau', type=float, default=0.2,
                        help='PPN only: weight of intrinsic-time regularization loss')
    parser.add_argument('--ppn_use_partition_correction', type=int, default=0,
                        help='PPN only: enable global-fit + partition-correction training (1/0)')
    parser.add_argument('--ppn_n_groups', type=int, default=1,
                        help='PPN only: N partitions for partition-correction stage')
    parser.add_argument('--ppn_m_cycles', type=int, default=1,
                        help='PPN only: repeat each partition pass for M cycles per epoch')
    parser.add_argument('--ppn_lambda_corr', type=float, default=0.1,
                        help='PPN only: weight of partition correction loss')
    parser.add_argument('--ppn_corr_margin', type=float, default=0.0,
                        help='PPN only: margin for correction hinge on per-step local/global MSE')
    parser.add_argument('--ppn_use_patch_embed', type=int, default=0,
                        help='PPN only: enable patch-level context embedding (1/0)')
    parser.add_argument('--ppn_patch_len', type=int, default=8,
                        help='PPN only: patch length for patch context embedding')
    parser.add_argument('--ppn_patch_stride', type=int, default=4,
                        help='PPN only: patch stride for patch context embedding')
    parser.add_argument('--ppn_use_var_tau_gate', type=int, default=0,
                        help='PPN only: enable variable-wise tau gating (1/0)')
    parser.add_argument('--ppn_var_tau_gate_scale', type=float, default=0.25,
                        help='PPN only: scale of variable-wise tau gate modulation')
    parser.add_argument('--ppn_use_multi_scale_tau', type=int, default=0,
                        help='PPN only: enable multi-scale intrinsic-time aggregation (1/0)')
    parser.add_argument('--ppn_tau_scales', type=str, default='1,2,4',
                        help='PPN only: comma-separated tau scales used by multi-scale aggregation')
    parser.add_argument('--ppn_use_tau_field', type=int, default=0,
                        help='PPN only: enable Version-A intrinsic time field (global+local) (1/0)')
    parser.add_argument('--ppn_tau_field_local_scale', type=float, default=0.5,
                        help='PPN only: scale for local component in tau field')
    parser.add_argument('--ppn_lambda_tau_smooth', type=float, default=0.0,
                        help='PPN only: weight of tau increment smoothness regularization')
    parser.add_argument('--ppn_use_tau_space_predictor', type=int, default=1,
                        help='PPN only: predict in tau space with explicit global mapping branch (1/0)')
    parser.add_argument('--ppn_tau_global_scale', type=float, default=1.0,
                        help='PPN only: scale for global mapping branch before tau-space adjustment')
    parser.add_argument('--ppn_tau_cross_adjust_scale', type=float, default=0.2,
                        help='PPN only: scale of cross adjustment between global and tau branches')
    parser.add_argument('--ppn_use_tau_cross_adjust_gate', type=int, default=1,
                        help='PPN only: gate the cross adjustment branch for stability (1/0)')
    parser.add_argument('--ppn_use_bidirectional_tau_coupling', type=int, default=0,
                        help='PPN only: enable bidirectional global/tau coupling gates (1/0)')
    parser.add_argument('--ppn_lambda_tau_recon', type=float, default=0.05,
                        help='PPN only: weight of tau-space reconstruction consistency loss')
    parser.add_argument('--ppn_lambda_tau_flat', type=float, default=0.01,
                        help='PPN only: weight of tau-space flatness (dynamics simplification) loss')
    parser.add_argument('--ppn_lambda_tau_mono', type=float, default=0.01,
                        help='PPN only: weight of monotonic clock constraint loss')
    parser.add_argument('--ppn_disable_accel_tau_loss_in_axiom_mode', type=int, default=1,
                        help='PPN only: disable acceleration-analogy tau loss when tau-space predictor is enabled (1/0)')
    parser.add_argument('--ppn_max_tau_integration_steps', type=int, default=0,
                        help='PPN only: maximum tau integration steps; 0 means model default')
    parser.add_argument('--ppn_use_transformer', type=int, default=0,
                        help='PPN only: use transformer for tau-step evolution instead of GRU (1/0)')
    parser.add_argument('--ppn_tau_stop_eps', type=float, default=1e-4,
                        help='PPN only: stop tolerance for tau-target integration')
    parser.add_argument('--ppn_tau_use_residual_refine', type=int, default=1,
                        help='PPN only: enable residual tau dynamics refine on top of GRU step (1/0)')
    parser.add_argument('--ppn_tau_coarse_kernel', type=int, default=4,
                        help='PPN only: kernel size for coarse tau clock branch')
    parser.add_argument('--ppn_tau_coarse_stride', type=int, default=4,
                        help='PPN only: stride for coarse tau clock branch')
    parser.add_argument('--ppn_lambda_tau_traj_smooth', type=float, default=0.002,
                        help='PPN only: weight of tau-trajectory smooth regularization')
    parser.add_argument('--ppn_lambda_proj_cycle', type=float, default=0.0,
                        help='PPN only: weight of bidirectional projection consistency')
    parser.add_argument('--ppn_lambda_tau_contrast', type=float, default=0.01,
                        help='PPN only: weight of tau-space contrastive alignment')
    parser.add_argument('--ppn_use_tau_phase_schedule', type=int, default=1,
                        help='PPN only: use 3-phase schedule for tau-space auxiliary losses (1/0)')
    parser.add_argument('--ppn_tau_phase1_ratio', type=float, default=0.3,
                        help='PPN only: fraction of epochs for phase-1 (global mapping focus)')
    parser.add_argument('--ppn_tau_phase2_ratio', type=float, default=0.5,
                        help='PPN only: fraction of epochs for phase-2 (tau geometry strengthening)')
    parser.add_argument('--ppn_tau_phase1_aux_scale', type=float, default=0.3,
                        help='PPN only: scale on tau auxiliary losses in phase-1')
    parser.add_argument('--ppn_tau_phase2_aux_scale', type=float, default=1.0,
                        help='PPN only: scale on tau auxiliary losses in phase-2')
    parser.add_argument('--ppn_tau_phase3_aux_scale', type=float, default=1.2,
                        help='PPN only: scale on tau auxiliary losses in phase-3')
    parser.add_argument('--ppn_use_horizon_residual', type=int, default=0,
                        help='PPN only: enable horizon-conditioned residual correction (1/0)')
    parser.add_argument('--ppn_use_horizon_residual_gate', type=int, default=0,
                        help='PPN only: enable lightweight gate on horizon residual correction (1/0)')
    parser.add_argument('--ppn_horizon_residual_scale', type=float, default=0.1,
                        help='PPN only: scale of horizon-conditioned residual correction')
    parser.add_argument('--ppn_use_horizon_residual_warmup', type=int, default=0,
                        help='PPN only: enable warmup schedule for horizon residual scale during training (1/0)')
    parser.add_argument('--ppn_horizon_residual_warmup_epochs', type=int, default=1,
                        help='PPN only: warmup epochs to reach full residual scale')
    parser.add_argument('--tau_base_model', type=str, default='DLinear',
                        help='TauFusion only: backbone model to fuse with tau mechanism')
    parser.add_argument('--tau_fuse_alpha', type=float, default=0.3,
                        help='TauFusion only: initial fusion weight for tau branch in [0,1]')
    parser.add_argument('--tauonly_anchor_delay_epochs', type=int, default=0,
                        help='TauOnly only: epochs to keep anchor branch disabled at training start')
    parser.add_argument('--tauonly_anchor_scale', type=float, default=1.0,
                        help='TauOnly only: base scale of anchor branch after delay')
    parser.add_argument('--tau_teacher_checkpoint', type=str, default='',
                        help='TauClock student only: checkpoint path for loading tau teacher weights')
    parser.add_argument('--tau_teacher_strict', type=int, default=1,
                        help='TauClock student only: strict teacher checkpoint loading (1/0)')
    parser.add_argument('--tau_teacher_trainable', type=int, default=0,
                        help='TauClock student only: whether tau teacher is trainable during student training (1/0)')
    parser.add_argument('--tau_student_hidden_dim', type=int, default=128,
                        help='TauClock student only: hidden dimension of weak GRU predictor')
    parser.add_argument('--tau_student_num_layers', type=int, default=1,
                        help='TauClock student only: number of GRU layers for weak predictor')
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='optional path to pretrained checkpoint file or directory')
    parser.add_argument('--load_checkpoint_strict', type=int, default=1,
                        help='whether to load pretrained checkpoint strictly (1/0)')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                        help='fraction of train set to use (0,1], useful for few-shot checks')
    parser.add_argument('--subset_seed', type=int, default=42,
                        help='random seed for train subset sampling when train_subset_ratio < 1.0')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    fix_seed = int(args.seed)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
