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

Ultra-high-resolution (UHR) remote sensing visual question answering (RS-VQA) requires interpreting fine-grained local details within their global spatial context under limited visual-token budgets.

**FocusEarth** is a training-free framework that couples **where to look** in the source image with **what to preserve** in the encoded visual representation. It leverages multiple frozen image–text encoders to localize question-relevant regions and establishes spatial correspondence between selected source regions and encoded visual features to guide visual token compression.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Overview of FocusEarth">
</p>

<p align="center">
  <em>Overview of the FocusEarth framework.</em>
</p>

### Framework

- **Query-Adaptive Region Prioritization.** Derives entity and context descriptions from the question and combines regional responses from multiple frozen image–text encoders through score standardization and entropy matching.

- **Multi-Query Region Selection.** Integrates context and entity proposals to select question-relevant regions under a shared region budget.

- **Spatially Aligned Visual Representation.** Maps selected source regions onto the encoded feature grid of a resized full-scene image, retaining their corresponding features individually while spatially aggregating the remaining features.

### Highlights

- **Training-free:** All pretrained image–text encoders and answer models remain frozen throughout inference.

- **Effective:** Achieves competitive performance across four UHR RS-VQA benchmarks, with accuracy gains across the evaluated frozen answer models.

- **Efficient:** Reduces visual-token count by **54.0%** and answering-stage latency by **49.0%** relative to direct Qwen3-VL-8B inference on MME-RealWorld-RS, with region scoring and selection precomputed offline.

This repository provides the official implementation of FocusEarth for ultra-high-resolution remote sensing visual question answering.

## Benchmarks

We evaluate FocusEarth on four representative UHR RS-VQA benchmarks covering diverse perception and reasoning tasks.

| Benchmark | Description | Questions | Accuracy (%) |
| :--- | :--- | ---: | ---: |
| [MME-RealWorld-RS](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) | Color, counting, and spatial position | 3,738 | 55.91 |
| [LRS-VQA](https://huggingface.co/datasets/ll-13/LRS-VQA) | Open-ended visual question answering | 7,333 | 34.32 |
| [XLRS-Bench-Lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite) | Perception and reasoning | 3,080 | 52.99 |
| [LRS-GRO](https://huggingface.co/datasets/HappyBug/LRS-GRO) | Global-, region-, and object-level VQA | 9,734 | 62.85 |

The overall accuracies are reported in our paper using Qwen3-VL-8B-Instruct as the answer model. Please refer to the respective dataset repositories for data access and preparation instructions.

## Installation

FocusEarth requires Python 3.12 and a CUDA-capable GPU for model inference. The package uses PyTorch 2.8.0 and Transformers 4.57.6.

```bash
# Clone the repository
git clone https://github.com/sun-wanghao/FocusEarth.git
cd FocusEarth

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install .

# Install the spaCy language model
python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

For detailed dependency and runtime requirements, see the [reproduction guide](docs/reproduction.md).

## Model Preparation

FocusEarth operates with frozen pretrained models without task-specific fine-tuning. Its applicability across different answer models is demonstrated in our paper, including Qwen3-VL, Qwen2.5-VL, LLaVA-v1.6, and Intern-S1-mini.

### Pretrained Checkpoints

The following checkpoints are used in the Qwen3-VL-8B-Instruct configuration:

| Model | Checkpoint | Source |
| :--- | :--- | :---: |
| CLIP | ViT-B/32 | [OpenAI](https://github.com/openai/CLIP) |
| GeoRSCLIP | `RS5M_ViT-B-32.pt` | [Hugging Face](https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32.pt) |
| RemoteCLIP | `RemoteCLIP-ViT-B-32.pt` | [Hugging Face](https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt) |
| SkyCLIP-50 | ViT-B/32, top 50%, epoch 20 | [Checkpoint](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip) |
| Qwen3-VL | Qwen3-VL-8B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) |

Set the local checkpoint paths under `scoring_encoders.checkpoints` and `answer_model.checkpoint` in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml).

For SkyCLIP-50, extract the downloaded archive and select the epoch-20 checkpoint. For Qwen3-VL, download the complete model snapshot, including the processor and tokenizer files.

See the [reproduction guide](docs/reproduction.md) for checkpoint specifications and SHA-256 checksums.

## Evaluation

The current release provides an evaluation entry point for MME-RealWorld-RS. The following example demonstrates the FocusEarth evaluation pipeline using Qwen3-VL-8B-Instruct.

### Data Preparation

Download `MME_RealWorld.json` and all `remote_sensing.tar.gz.part_*` archives from [MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld). Merge and extract the image archives following the [official instructions](https://github.com/MME-Benchmarks/MME-RealWorld).

Organize the dataset as follows, preserving the original relative image paths:

```text
data/
└── MME-RealWorld-RS/
    ├── MME_RealWorld.json
    └── remote_sensing/
        └── ...
```

Update the dataset paths in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml):

```yaml
dataset:
  annotations: /path/to/MME-RealWorld-RS/MME_RealWorld.json
  root: /path/to/MME-RealWorld-RS
```

The evaluation script automatically selects the Remote Sensing subset from the original annotations.

### Example Usage

After configuring the model checkpoints and dataset paths, run the following command from the repository root:

```bash
python scripts/eval_mme_realworld_rs.py \
  --config configs/mme_realworld_rs.yaml
```

The answer model is prompted to provide an image-grounded evidence description followed by a final option letter:

```text
Evidence: [A concise, image-grounded observation-based explanation]
Final answer: [A single MME option letter]
```

Evaluation reports overall and per-category accuracy for Color, Count, and Position. Invalid predictions and execution failures are counted as incorrect.

### Outputs

The evaluation script saves the following files to the configured output directory:

| File | Description |
| :--- | :--- |
| `predictions.jsonl` | Model responses and answer predictions |
| `traces.jsonl` | Region rankings, selected regions, and visual-token statistics |
| `results.json` | Overall and per-category evaluation results |
| `run_manifest.json` | Runtime configuration and model specifications |

For detailed evaluation settings and runtime requirements, see the [reproduction guide](docs/reproduction.md).

## Documentation

- [Implementation Guide](docs/implementation.md): Detailed implementation of the FocusEarth framework.
- [Reproduction Guide](docs/reproduction.md): Model checkpoints, runtime environment, and evaluation settings.

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you find FocusEarth useful in your research, please cite our work:

```bibtex
@misc{sun2026focusearth,
  title   = {FocusEarth: Where to Look and What to Preserve for Ultra-High-Resolution Remote Sensing {VQA}},
  author  = {Sun, Wanghao and Liu, Nanqing and Sun, Yu and Hu, Zhanxuan and Ming, Qi and Li, Heng-Chao},
  year    = {2026}
}
```

## Acknowledgements

We thank the teams behind [CLIP](https://github.com/openai/CLIP), [GeoRSCLIP](https://github.com/om-ai-lab/RS5M), [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), [SkyCLIP](https://github.com/wangzhecheng/SkyScript), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for making their pretrained models accessible.

We are grateful to the creators of [MME-RealWorld](https://github.com/MME-Benchmarks/MME-RealWorld), [LRS-VQA](https://huggingface.co/datasets/ll-13/LRS-VQA), [XLRS-Bench](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite), and [LRS-GRO](https://huggingface.co/datasets/HappyBug/LRS-GRO) for providing the benchmarks used in our research.

We also thank the developers of [OpenCLIP](https://github.com/mlfoundations/open_clip), [Transformers](https://github.com/huggingface/transformers), and [spaCy](https://github.com/explosion/spaCy) for their open-source contributions.

External models and datasets are subject to their respective licenses and terms.
