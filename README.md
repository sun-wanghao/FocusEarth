<h1 align="center">
FocusEarth: Where to Look and What to Preserve<br>
for Ultra-High-Resolution Remote Sensing VQA
</h1>

<p align="center">
  Wanghao Sun<sup>1</sup>,
  Nanqing Liu<sup>1,*</sup>,
  Yu Sun<sup>1</sup>,
  Zhanxuan Hu<sup>1</sup>,
  Qi Ming<sup>2</sup>,
  Heng-Chao Li<sup>3</sup>
</p>

<p align="center">
  <sup>1</sup> Yunnan Normal University &nbsp;
  <sup>2</sup> Beijing University of Technology &nbsp;
  <sup>3</sup> Southwest Jiaotong University
  <br>
  <sub>* Corresponding author</sub>
</p>

## Overview

**FocusEarth** is a training-free framework that couples **where to look** in the source image with **what to preserve** in the encoded visual representation for ultra-high-resolution remote sensing visual question answering (UHR RS-VQA).

FocusEarth combines multi-encoder regional relevance estimation with entity- and context-guided region selection. The selected regions guide feature-level protection and spatial aggregation within a resized full-scene image, retaining question-relevant visual features while maintaining the global spatial layout.

Experiments on four UHR RS-VQA benchmarks demonstrate competitive performance across multiple frozen answer models. With Qwen3-VL-8B, FocusEarth reduces visual-token count by **54.0%** and answering-stage latency by **49.0%** on MME-RealWorld-RS relative to direct inference, with region scoring and selection precomputed offline.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Overview of FocusEarth">
</p>

## Benchmarks

| Benchmark | Questions | Accuracy (%) |
| :--- | ---: | ---: |
| [MME-RealWorld-RS](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) | 3,738 | 55.91 |
| [LRS-VQA](https://huggingface.co/datasets/ll-13/LRS-VQA) | 7,333 | 34.32 |
| [XLRS-Bench-Lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite) | 3,080 | 52.99 |
| [LRS-GRO](https://huggingface.co/datasets/HappyBug/LRS-GRO) | 9,734 | 62.85 |

Results reported in our paper using Qwen3-VL-8B-Instruct as the answer model.

## Installation

FocusEarth requires Python 3.12 and a CUDA-capable GPU for inference.

```bash
git clone https://github.com/sun-wanghao/FocusEarth.git
cd FocusEarth

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install .

python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Models

Download the pretrained checkpoints for the Qwen3-VL-8B-Instruct configuration:

| Model | Checkpoint | Role |
| :--- | :--- | :--- |
| CLIP | [ViT-B/32](https://github.com/openai/CLIP) | Region scoring |
| GeoRSCLIP | [ViT-B/32](https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32.pt) | Region scoring |
| RemoteCLIP | [ViT-B/32](https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt) | Region scoring |
| SkyCLIP-50 | [ViT-B/32, Top 50%, Epoch 20](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip) | Region scoring |
| Qwen3-VL | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | Answer model |

Use the epoch-20 checkpoint for SkyCLIP-50 and the complete model snapshot for Qwen3-VL. Set the local checkpoint paths in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml).

See the [reproduction guide](docs/reproduction.md) for checkpoint specifications and environment requirements.

## Evaluation

The released evaluation pipeline is illustrated below using MME-RealWorld-RS.

### Data Preparation

Download `MME_RealWorld.json` and the `remote_sensing.tar.gz.part_*` archives from [MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld). Extract the images following the [official instructions](https://github.com/MME-Benchmarks/MME-RealWorld).

```text
data/
└── MME-RealWorld-RS/
    ├── MME_RealWorld.json
    └── remote_sensing/
        └── ...
```

Configure the dataset paths in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml):

```yaml
dataset:
  annotations: /path/to/MME-RealWorld-RS/MME_RealWorld.json
  root: /path/to/MME-RealWorld-RS
```

### Example Usage

```bash
python scripts/eval_mme_realworld_rs.py \
  --config configs/mme_realworld_rs.yaml
```

Outputs include `predictions.jsonl`, `traces.jsonl`, `results.json`, and `run_manifest.json`.

For further details, see the [implementation guide](docs/implementation.md) and [reproduction guide](docs/reproduction.md).

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{sun2026focusearth,
  title={FocusEarth: Where to Look and What to Preserve for Ultra-High-Resolution Remote Sensing {VQA}},
  author={Sun, Wanghao and Liu, Nanqing and Sun, Yu and Hu, Zhanxuan and Ming, Qi and Li, Heng-Chao},
  year={2026}
}
```

## Acknowledgements

## Acknowledgements

- Thanks to the authors of [ZoomSearch](https://github.com/kiki-zyq/ZoomSearch) for their excellent work.
- Thanks to the authors of [WeaveEarth](https://github.com/XianZhi-Ma/WeaveEarth) for their excellent work.
