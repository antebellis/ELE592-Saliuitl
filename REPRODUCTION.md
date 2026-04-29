# Quick Run Guide (Base + Extensions)

This repo supports:
- Base Saliuitl runs (as in upstream docs)
- APRICOT preparation and evaluation
- Real-time overhead extension (mode/FPS comparison)

## 1) Environment

```powershell
cd "C:\path\to\ELE592-Saliuitl\Project"
py -3.12 -m venv .venv312
.\.venv312\Scripts\python -m pip install --upgrade pip setuptools wheel
```

Install dependencies on Windows:

```powershell
$lines = Get-Content requirements.txt | Where-Object {$_ -notmatch '^nvidia-nccl-cu11==' -and $_ -notmatch '^triton=='}
Set-Content -Path requirements.windows.txt -Value $lines
.\.venv312\Scripts\python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.windows.txt
```

## 2) Required Weights

YOLOv2 weights are required for INRIA/VOC/APRICOT runs:

```powershell
New-Item -ItemType Directory -Force -Path weights | Out-Null
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov2.weights" -OutFile "weights/yolo.weights"
```

## 3) Base Saliuitl (example)

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1
```

Clean counterpart:

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1 --clean
```

## 4) APRICOT Dataset Preparation

1. Download/extract APRICOT so `APRICOTv1.0/` contains `Images/` and `Annotations/`.
2. Build Saliuitl-format APRICOT folders:

```powershell
.\.venv312\Scripts\python prepare_apricot_for_saliuitl.py --apricot_root APRICOTv1.0 --out_root data/apricot_saliuitl --size 416 --pad 6
```

This creates:
- `data/apricot_saliuitl/1p` (attacked frames + `effective_1p.npy`)
- `data/apricot_saliuitl/clean` (inpainted clean counterpart)

## 5) APRICOT Evaluation (base Saliuitl)

```powershell
.\.venv312\Scripts\python saliuitl.py --inpaint biharmonic --imgdir data/apricot_saliuitl/clean --patch_imgdir data/apricot_saliuitl/1p --dataset voc --det_net_path checkpoints/final_detection/2dcnn_raw_VOC_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1
```

## 6) Real-Time Overhead Extension

Compares 4 modes:
- classification-only
- detection-only
- detection+recovery
- always-recovery

```powershell
.\.venv312\Scripts\python realtime_mode_overhead.py --out_root realtime_mode_overhead_apricot_full --dataset voc --imgdir data/apricot_saliuitl/clean --patch_imgdir data/apricot_saliuitl/1p --det_net_path checkpoints/final_detection/2dcnn_raw_VOC_5_atk_det.pth --ensemble_step 5 --inpainting_step 5
```

No-reuse (blind recovery timing):

```powershell
.\.venv312\Scripts\python realtime_mode_overhead.py --out_root realtime_mode_overhead_apricot_full_s5_noreuse --dataset voc --imgdir data/apricot_saliuitl/clean --patch_imgdir data/apricot_saliuitl/1p --det_net_path checkpoints/final_detection/2dcnn_raw_VOC_5_atk_det.pth --ensemble_step 5 --inpainting_step 5 --force_no_reuse
```

## 7) Visuals

Create slide visuals from summaries:

```powershell
.\.venv312\Scripts\python make_apricot_slide_visuals.py
.\.venv312\Scripts\python make_apricot_eval_visuals.py
.\.venv312\Scripts\python make_apricot_sample_visual.py
```

