# Reproduction guide

## Environment

Install the package and `en_core_web_sm` 3.8.0 as shown in the [README](../README.md#installation). Python dependencies are pinned in [pyproject.toml](../pyproject.toml) and [requirements.txt](../requirements.txt). Transformers 4.57.6 is required because the visual integration uses its Qwen3-VL feature and rotary-position interfaces. Accelerate 1.12.0 provides automatic device placement when loading Qwen.

## Checkpoints

From the repository root, run `python scripts/download_models.py` after installing the Python dependencies. It downloads from the original model sources, verifies the four encoder SHA-256 hashes, and retrieves the complete Qwen snapshot at the pinned revision using `huggingface_hub` (installed by Transformers). No GPU is required. Qwen file sizes and Git/LFS content identities are checked against the metadata for that revision. The downloader is separate from evaluation; evaluation does not download models automatically.

The default local paths are:

```text
models/
├── CLIP/ViT-B-32.pt
├── GeoRSCLIP/RS5M_ViT-B-32.pt
├── RemoteCLIP/RemoteCLIP-ViT-B-32.pt
├── SkyCLIP-50/epoch_20.pt
└── Qwen3-VL-8B-Instruct/
```

To prepare a single model, pass `--only CLIP`, `--only GeoRSCLIP`, `--only RemoteCLIP`, `--only SkyCLIP-50`, or `--only Qwen`. Pass `--config PATH` for a custom configuration. Run `python scripts/download_models.py --check-only` to verify local checkpoints without downloading weights (Qwen verification requires online metadata). Alternatively, obtain the files manually from the links in the [README](../README.md#models) and use the same directory layout.

Rerunning reuses verified checkpoints. HTTP partial files are resumed if the server supports byte ranges; otherwise they restart. Hugging Face manages partial Qwen downloads in its local cache. The SkyCLIP archive is retained for reuse and only its unique `epoch_20.pt` member is extracted. If a file fails verification, move the named incorrect file aside and rerun; the script does not replace an existing invalid checkpoint automatically. Use a dedicated Qwen directory containing only the pinned snapshot files and the Hugging Face cache. Do not run concurrent preparation processes targeting the same paths. The Qwen snapshot alone is approximately 17.5 GB; allow additional space for encoder weights, the SkyCLIP archive, temporary downloads and cache overhead. Free space is checked before transfers and extraction where sizes are available.

The default encoder paths are defined in [the MME configuration](../configs/mme_realworld_rs.yaml) under `scoring_encoders.checkpoints`. Files are verified against these SHA-256 identities:

| Encoder | SHA-256 |
| --- | --- |
| CLIP ViT-B/32 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| GeoRSCLIP RS5M ViT-B/32 | `129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01` |
| RemoteCLIP ViT-B/32 | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` |
| SkyCLIP-50 ViT-B/32 | `69ba9fa7089322a83e9164fe515a222bcece2a715888db5749ab5ce4d1602313` |

By default, `answer_model.checkpoint` points to `models/Qwen3-VL-8B-Instruct/`. It must identify a complete local `Qwen/Qwen3-VL-8B-Instruct` snapshot at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Keep its processor, tokenizer, model configuration, and weight files together. The model preparation script verifies the pinned snapshot against upstream metadata; the existing inference loader itself does not independently verify the Qwen file hashes. Checkpoint loading uses pickle-capable PyTorch APIs where required by the file format; use trusted files.

## Dataset and configuration

The default dataset root is `data/MME-RealWorld-RS/`, with annotations at `data/MME-RealWorld-RS/MME_RealWorld.json`. `dataset.annotations` points to the official MME-RealWorld JSON list. `dataset.root` contains the image paths named by each row's `Image` field. Required fields are `Question_id`, `Subtask`, `Category`, `Image`, `Text`, `Answer choices`, and `Ground truth`. Remote sensing categories use lowercase `color`, `count`, and `position`; the five original option strings retain `(A)` through `(E)` labels, including blank option content.

The configuration groups external inputs separately from method settings. Only paths, the scoring device, and scoring image batch size are user settings. Region-prioritization, selection, visual-representation, model, and generation values are checked against the implementation. Paths are resolved relative to the working directory; `~` is expanded. Dataset and output CLI arguments override their YAML paths.

## Execution and outputs

Run the [README evaluation command](../README.md#evaluation) from the repository root. The default configuration needs no edits when the resources follow the documented layout. `python scripts/eval_mme_realworld_rs.py --help` lists the configuration, dataset, annotation, and output arguments. The default output directory is `outputs/mme_realworld_rs/`; it must not already exist. Use `--output-dir` to select a fresh directory for each subsequent run.

The command scores all questions before loading the answering model. Encoder models are cached across scoring questions and released before Qwen loading. A private intermediate file in the output directory transfers region plans between stages and is removed when answering finishes. No precomputed score bank is required.

`predictions.jsonl` preserves raw responses and errors. `traces.jsonl` records descriptions, regions, geometry, protection, grouping identity, and effective budgets without copying source images. `results.json` pools correct/total counts over all 3,738 questions and each category; invalid answers and execution failures are incorrect. `run_manifest.json` records the effective settings, annotation and scoring-checkpoint hashes, and dependency versions, using path basenames rather than machine-specific directories. Interrupted runs are not resumed automatically.

## Checks and runtime requirements

Run `python -m unittest discover -s tests -v` for lightweight geometry, fusion, selection, query-transformation, parser, metric, configuration, and model-interface checks. Interface fixtures do not load checkpoints. These tests do not establish encoder-forward equivalence, generated-answer accuracy, or pixel-level equivalence to the external image-fetch helper.

The image adapter applies factor-28 RGB bicubic resizing before the pinned processor. End-to-end inference requires the four scoring checkpoints, the spaCy language model, Qwen weights, source images, and suitable GPU memory. Offline four-encoder scoring contributes to total computation and is excluded from answering-stage latency.
