# UniSVQ & LC-QAT: 2-Bit LLM Quantization with Linear-Constrained Vector Quantization

This repository contains the official implementations of two companion papers on 2-bit LLM quantization, along with further improvements for efficient inference:

- **UniSVQ** ([arXiv:2606.10520](https://arxiv.org/abs/2606.10520)): A **post-training quantization (PTQ)** framework that unifies scalar and vector quantization. It parameterizes codewords as an affine transform of integer lattices, preserving compatibility with optimized integer kernels while retaining the flexibility of VQ.
- **LC-QAT** ([arXiv:2606.10531](https://arxiv.org/abs/2606.10531)): A data-efficient 2-bit **quantization-aware training (QAT)** framework. It introduces linear-constrained vector quantization (LCVQ), replacing discrete codebook lookup with a learned linear projection for end-to-end differentiable training. 

> **Further improvements over the papers:** We further replace the original codebook with a **linear codebook** and constrain the Hadamard transform to operate on **128×128 blocks** (instead of full-dimension), enabling fast CUDA-accelerated inference via our `hadaquant` kernel.

This repository provides the code for training post-training quantization (PTQ) and quantization-aware training (QAT) on Qwen3 1.7B. Building on the experiments in the papers, we expanded the training set to approximately 8B tokens and used the [UltraFineWeb-Edu dataset](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) to further improve model performance. The performance of our trained 2-bit model on Qwen3 1.7B is shown below:

| model | ARC-C | ARC-E | BoolQ | HellaSwag | PIQA | WinoGrande | Avg. | per |
|-------|-------|-------|-------|-----------|------|------------|------|-----|
| FP16  | 43.08 | 69.69 | 77.52 | 60.37     | 72.14 | 61.80      | 64.10 | 1.00 |
| 2bit  | 42.32 | 71.04 | 68.99 | 61.04     | 73.72 | 62.90      | 63.34 | 0.99 |

We also strengthen the model's performance on math, code, and complex tasks using approximately 8B tokens of conversational supervised data. The performance of our 2-bit model after 8B supervised fine-tuning on Qwen3 1.7B is shown below:

The model is available at our [🤗 huggingface repo](https://huggingface.co/w-hy21/lcqat_qwen3_1.7B).

| Model              | OpenbookQA | If    | MMLU  | GSM8K | MATH  | HumanEval | BBH   | Avg.  |
|-----------------|------------|-------|-------|-------|-------|-----------|-------|-------|
| Qwen3 1.7B fp16 | 64.40      | 74.34 | 63.87 | 83.70 | 71.20 | 60.98     | 60.47 | 68.42 |
| BitNet 2B4T     | 41.60      | 53.48 | 53.17 | 58.38 | 43.40 | 38.40     | 49.83 | 48.32 |
| 2bit  | 66.00      | 71.94 | 58.98 | 70.36 | 42.20 | 46.95     | 52.58 | 58.43 |


## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch`, `flash-attn`, and `deepspeed` require a CUDA toolkit. If installation fails, install them manually according to your CUDA version:
> ```bash
> pip install torch==2.9.0
> pip install flash-attn==2.8.3
> pip install deepspeed
> ```

Key dependencies:
- `torch==2.9.0`
- `ms-swift>=4.3.1`
- `flash-attn==2.8.3`
- `lm_eval`
- `fast-hadamard-transform`

### 2. Install HadaQuant CUDA Extension

The HadaQuant package provides optimized CUDA kernels for 128-block Hadamard transforms and packed 2-bit dequantization:

```bash
cd hadaquant/csrc
pip install -e .
cd ../..
```

The kernels are compiled for SM 80/86/89/90 (A100, RTX 3090, RTX 4090, H100). Edit `setup.py` to add/remove GPU architectures as needed.

## Pipeline Overview

The full pipeline consists of 3 stages + evaluation:

1. **PTQ Initialization** — `unisvq_qwen3_1.7B.sh`
   - Compute Hessians for optimal quantization
   - Quantize weights and fine-tune for reconstruction
   - Convert to HuggingFace format (HFize)

2. **Checkpoint Conversion** — `scripts/create_qwen3_ckpt_from_quip.py`
   - Convert PTQ weights to ms_swift trainable checkpoint
   - Apply block-wise Hadamard transforms on 128×128 blocks
   - Initialize linear codebook parameters

3. **QAT Training** — `lc_qat_1.7B.sh`
   - Distributed QAT fine-tuning with ms_swift + DeepSpeed

4. **Evaluation**
   - LM benchmarks: `lm_eval.sh`
   - Inference throughput: `quantized_infer.py` (requires `hadaquant` CUDA extension)

## Usage

### Step 1: PTQ Initialization

Edit `unisvq_qwen3_1.7B.sh` and set the paths:

```bash
base_model_path=YOUR_BASE_MODEL_PATH    # e.g., path to Qwen3-1.7B
exp_dir=YOUR_EXP_DIR                    # Output directory
dataset_path="YOUR_DATASET_PATH/pajama.jsonl"  # Calibration dataset
```

Our calibration dataset is sampled from `togethercomputer/RedPajama-Data-1T-Sample`.

The script runs in 3 stages (controlled by `stage` variable):
- **Stage 1** (`stage=1`): Compute Hessians for optimal quantization
- **Stage 2** (`stage=2`): Quantize weights and fine-tune for reconstruction
- **Stage 3** (`stage=3`): Convert to HuggingFace format

```bash
bash unisvq_qwen3_1.7B.sh
```

### Step 2: Convert Checkpoint for QAT

Convert the PTQ checkpoint into a format suitable for QAT training with ms_swift:

```bash
python scripts/create_qwen3_ckpt_from_quip.py \
    --ori_model_path YOUR_BASE_MODEL_PATH \
    --init_ckpt_path ${exp_dir}/hf${exp_suffix} \
    --output_path ${exp_dir}/Qwen3-1.7B-pretrain/init_ckpt
```

This script:
1. Loads the PTQ quantized weights and scales (SU, SV, codebook)
2. Initializes the trainable weights for the linear codebook
3. Saves the checkpoint in ms_swift-compatible format

### Step 3: QAT Training

Edit `lc_qat_1.7B.sh` and set the paths, then run:

```bash
bash lc_qat_1.7B.sh
```

This uses `ms_swift` for distributed QAT fine-tuning with DeepSpeed. The training configuration is specified in `configs/1.7B.yaml` (ms_swift format). Key settings:
- 4 GPUs by default (`NPROC_PER_NODE=4`)
- Uses the `qwen3_lcqat` model type registered in `model/register.py`

### Step 4: Evaluation

#### LM Evaluation Benchmark

Edit `lm_eval.sh` to set `exp_dir` and `origin_model_path`, then:

```bash
bash lm_eval.sh
```

This unpacks the QAT checkpoint and evaluates on standard benchmarks: ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, PIQA, and WinoGrande.

#### Inference Throughput Evaluation

After installing the HadaQuant CUDA extension:

```bash
./quantized_infer.sh
```

## Key Technical Details

### Linear Codebook (LCVQ)

Traditional vector quantization uses a discrete codebook lookup, which requires a straight-through estimator (STE) to pass gradients. Our linear codebook ([`lib/codebook/index_codebook.py`](lib/codebook/index_codebook.py)) replaces this with an orthogonal linear projection using a fixed orthogonal matrix. This achieves better performance than conventional 2-bit scalar methods while allowing end-to-end QAT training of all quantization weights.

### Block-wise Hadamard Transform

The original method applies Hadamard transforms over the full input/output dimensions, which is expensive for inference. We constrain the transform to **128×128 blocks**, enabling CUDA kernel fusion of the Hadamard transform with SU/SV scaling. This block-wise design also supports tensor parallelism for distributed training.

### Differentiable Quantization (DGE)

During QAT training, we adopt the [differentiable gradient estimator](https://arxiv.org/abs/2501.17116) proposed by Ruizhe Wang et al. in *Optimizing Large Language Model Training Using FP4 Quantization*. We further introduce stochastic gradient masking to stabilize training, and enable end-to-end training of the codebook projection, scales (SU/SV), and latent weights.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026lcqat,
  title   = {LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization},
  author  = {Wang, Haoyu and Yu, Xingyu and Zhao, Haiyan and Wang, Fengxiang and Han, Xu},
  journal = {arXiv preprint arXiv:2606.10531},
  year    = {2026}
}

@article{wang2026unisvq,
  title   = {UniSVQ: 2-bit Unified Scalar-Vector Quantization},
  author  = {Wang, Haoyu and Zhao, Haiyan and Yu, Xingyu and Yao, Zhangyang and Han, Xu and Liu, Zhiyuan and Sun, Maosong},
  journal = {arXiv preprint arXiv:2606.10520},
  year    = {2026}
}
```
