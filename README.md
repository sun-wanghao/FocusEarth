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

**FocusEarth** is a training-free framework that couples **where to look** in the source image with **what to preserve** in the encoded visual representation. It leverages multiple frozen image–text encoders to localize question-relevant regions and uses their spatial correspondence to guide visual token compression.

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

- **Training-free:** All image–text encoders and answer models remain frozen throughout inference.
- **Effective:** Achieves competitive performance across four UHR RS-VQA benchmarks.
- **Efficient:** Reduces visual-token count by **54.0%** and answering-stage latency by **49.0%** with Qwen3-VL-8B relative to direct inference on MME-RealWorld-RS, with region scoring and selection precomputed offline.

This repository provides the official implementation of FocusEarth, including inference and evaluation on MME-RealWorld-RS.


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

FocusEarth uses four frozen image–text encoders for region prioritization and Qwen3-VL-8B-Instruct as the answer model.

Download the pretrained checkpoints from the following sources:

| Model | Checkpoint | Download |
| :--- | :--- | :---: |
| CLIP | ViT-B/32 | [OpenAI](https://github.com/openai/CLIP) |
| GeoRSCLIP | `RS5M_ViT-B-32.pt` | [Hugging Face](https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32.pt) |
| RemoteCLIP | `RemoteCLIP-ViT-B-32.pt` | [Hugging Face](https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt) |
| SkyCLIP-50 | ViT-B/32, top 50%, epoch 20 | [Checkpoint](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip) |
| Qwen3-VL | Qwen3-VL-8B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) |

Set the local checkpoint paths in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml):

| Model | Configuration key |
| :--- | :--- |
| CLIP | `scoring_encoders.checkpoints.CLIP` |
| GeoRSCLIP | `scoring_encoders.checkpoints.GeoRSCLIP` |
| RemoteCLIP | `scoring_encoders.checkpoints.RemoteCLIP` |
| SkyCLIP-50 | `scoring_encoders.checkpoints.SkyCLIP-50` |
| Qwen3-VL | `answer_model.checkpoint` |

For SkyCLIP-50, extract the downloaded archive and use `epoch_20.pt`. For Qwen3-VL, download the complete model snapshot, including the processor and tokenizer files.

See the [reproduction guide](docs/reproduction.md) for checkpoint specifications and SHA-256 checksums.

## Dataset Preparation

Download [MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) from the official Hugging Face repository. We use the Remote Sensing subset (MME-RealWorld-RS), which contains 3,738 multiple-choice questions covering Color, Count, and Position.

Download `MME_RealWorld.json` and all `remote_sensing.tar.gz.part_*` archives. Merge and extract the image archives following the [official instructions](https://github.com/MME-Benchmarks/MME-RealWorld).

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
  annotations: data/MME-RealWorld-RS/MME_RealWorld.json
  root: data/MME-RealWorld-RS
```

The evaluation script automatically selects the Remote Sensing subset from the original annotations.

## Evaluation

### MME-RealWorld-RS

After preparing the model checkpoints and dataset, configure the corresponding paths in [`configs/mme_realworld_rs.yaml`](configs/mme_realworld_rs.yaml).

Run the following command from the repository root:

FocusEarth performs query-adaptive region prioritization, multi-query region selection, and spatially aligned visual representation construction before answer generation.

The answer model generates an image-grounded evidence description followed by a final option letter:

```text
Evidence: [A concise, image-grounded observation-based explanation]
Final answer: [A single MME option letter]
```

Evaluation reports overall and per-category accuracy for Color, Count, and Position. Invalid predictions and execution failures are counted as incorrect.

### Outputs

The evaluation script saves the following files to the specified output directory:

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

## Acknowledgements

We thank the teams behind [CLIP](https://github.com/openai/CLIP), [GeoRSCLIP](https://github.com/om-ai-lab/RS5M), [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), [SkyCLIP](https://github.com/wangzhecheng/SkyScript), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for making their pretrained models accessible.

We also thank the creators of [MME-RealWorld](https://github.com/MME-Benchmarks/MME-RealWorld) and the developers of [OpenCLIP](https://github.com/mlfoundations/open_clip), [Transformers](https://github.com/huggingface/transformers), and [spaCy](https://github.com/explosion/spaCy) for their valuable contributions to the research community.

External models and datasets are subject to their respective licenses and terms.
