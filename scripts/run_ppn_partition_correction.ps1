$ErrorActionPreference = 'Stop'

$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUNBUFFERED = '1'

$dataset = 'electricity'
$dataPath = '.\\datasets\\tslib_hf\\electricity\\electricity.csv'
$predLens = @(96, 192)
$subsetRatios = @(1.0, 0.1)

foreach ($predLen in $predLens) {
  foreach ($ratio in $subsetRatios) {
    $tag = "ppn_nm_${dataset}_h${predLen}_r${ratio}"
    Write-Host "[RUN] $tag"

    python .\\run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --model_id $tag `
      --model PPN `
      --data custom `
      --root_path .\\datasets\\tslib_hf\\electricity\\ `
      --data_path electricity.csv `
      --features M `
      --seq_len 96 `
      --label_len 48 `
      --pred_len $predLen `
      --enc_in 321 `
      --dec_in 321 `
      --c_out 321 `
      --d_model 64 `
      --e_layers 2 `
      --d_layers 1 `
      --d_ff 128 `
      --batch_size 16 `
      --train_epochs 8 `
      --patience 3 `
      --learning_rate 5e-4 `
      --lradj cosine `
      --des nm_partition `
      --itr 1 `
      --use_gpu `
      --gpu_type cuda `
      --gpu 0 `
      --ppn_use_tau_loss 1 `
      --ppn_lambda_tau 0.05 `
      --ppn_use_partition_correction 1 `
      --ppn_n_groups 4 `
      --ppn_m_cycles 2 `
      --ppn_lambda_corr 0.2 `
      --ppn_corr_margin 0.0 `
      --train_subset_ratio $ratio `
      --subset_seed 42
  }
}

Write-Host '[DONE] partition-correction experiments finished.'
