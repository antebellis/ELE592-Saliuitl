# Saliuitl Reproduction (Windows + CUDA)

This project reproduces the official `Saliuitl` repository in `Project/` and runs the README commands on the bundled INRIA subset.

## 1) Environment setup

```powershell
cd "C:\Users\ndhan\PycharmProjects\ELE 592\Project"
py -3.12 -m venv .venv312
.\.venv312\Scripts\python -m pip install --upgrade pip setuptools wheel
```

The upstream `requirements.txt` contains Linux-only packages (`nvidia-nccl-cu11`, `triton`) that do not install on Windows. Use:

```powershell
$lines = Get-Content requirements.txt | Where-Object {$_ -notmatch '^nvidia-nccl-cu11==' -and $_ -notmatch '^triton=='}
Set-Content -Path requirements.windows.txt -Value $lines
.\.venv312\Scripts\python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.windows.txt
```

## 2) Download YOLOv2 weights (required for INRIA/VOC)

```powershell
New-Item -ItemType Directory -Force -Path weights | Out-Null
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov2.weights" -OutFile "weights/yolo.weights"
```

## 3) Run official README commands

Single rectangular patch on INRIA:

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1
```

Observed output on current bundled subset (30 images):

- `Unsuccesful Attacks: 0.7916666666666666`
- `Detected Attacks: 0.9583333333333334`
- `Successful Attacks: 0.20833333333333334`

Clean counterpart (`--clean`):

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1 --clean
```

Observed output:

- `Unsuccesful Attacks: 1.0`
- `Detected Attacks: 0.0`
- `Successful Attacks: 0.0`

Double rectangular patches:

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/2p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_2p.npy --n_patches 2
```

Observed output:

- `Unsuccesful Attacks: 0.8571428571428571`
- `Detected Attacks: 1.0`
- `Successful Attacks: 0.14285714285714285`

## 4) Notes for full paper-level replication

- The repository includes a reduced example subset; paper tables are computed on larger evaluation sets.
- CIFAR commands additionally require `checkpoints/resnet50_192_cifar.pth` from PatchGuard:
  - https://github.com/inspire-group/PatchGuard/tree/master
- VOC/INRIA need YOLOv2 weights in `weights/yolo.weights` (step 2 above).
