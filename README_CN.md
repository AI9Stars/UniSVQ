# UniSVQ & LC-QAT：基于线性约束向量量化的 2-Bit LLM 量化方法

本仓库包含以下两篇 2-bit LLM 量化论文的官方实现，并在此基础上做了进一步改进以支持高效推理：

- **UniSVQ** ([arXiv:2606.10520](https://arxiv.org/abs/2606.10520))：一种统一的 2-bit **后训练量化（PTQ）** 框架，将码字参数化为整数格的仿射变换，兼顾了标量量化的推理效率与向量量化的精度优势。
- **LC-QAT** ([arXiv:2606.10531](https://arxiv.org/abs/2606.10531))：一种数据高效的 2-bit **量化感知训练（QAT）** 框架，提出线性约束向量量化（LCVQ），将传统离散码本查表替换为可学习的线性投影，实现端到端可微分训练。

> **相比论文的进一步改进：** 我们将原始码本进一步替换为**线性码本**，并将 Hadamard 变换限制在 **128×128 的块**上（而非全维度），从而支持通过 `hadaquant` CUDA 内核实现快速推理。

此开源代码提供了我们在Qwen3 1.7B上进行训练后量化和量化感知训练的代码。在论文实验的基础上，我们将训练集扩展到了约8B Token，并使用[ultrafineweb-edu数据集](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)进一步提升模型表现。在Qwen3 1.7B上，我们训练后的2bit模型的性能表现如下：

|model|ARC-C|ARC-E|BoolQ|HellaSwag|PIQA|WinoGrande|Avg.|per|
|-----|-----|-----|-----|---------|----|----------|----|---|
|FP16|43.08|69.69|77.52|60.37|72.14|61.80|64.10|1.00|
|2bit|42.32|71.04|68.99|61.04|73.72|62.90|63.34|0.99|

我们还进一步使用约8B的对话形式有监督数据强化了模型在数学、代码和复杂任务上的表现。在Qwen3 1.7B上，我们通过8B有监督数据训练后的2bit模型的性能表现如下：

模型可在[🤗 我们的 huggingface 仓库](https://huggingface.co/w-hy21/lcqat_qwen3_1.7B)上获取。

| 模型              | OpenbookQA | If    | MMLU  | GSM8K | MATH  | HumanEval | BBH   | Avg.  |
|-----------------|------------|-------|-------|-------|-------|-----------|-------|-------|
| Qwen3 1.7B fp16 | 64.40      | 74.34 | 63.87 | 83.70 | 71.20 | 60.98     | 60.47 | 68.42 |
| BitNet 2B4T     | 41.60      | 53.48 | 53.17 | 58.38 | 43.40 | 38.40     | 49.83 | 48.32 |
| 2bit  | 66.00      | 71.94 | 58.98 | 70.36 | 42.20 | 46.95     | 52.58 | 58.43 |



## 安装

### 1. 安装 Python 依赖

**使用 pip：**

```bash
pip install -r requirements.txt
```

> **注意：** `torch`、`flash-attn`、`deepspeed` 需要 CUDA 环境。如果安装失败，请根据你的 CUDA 版本手动安装：
> ```bash
> pip install torch==2.9.0
> pip install flash-attn==2.8.3
> pip install deepspeed
> ```

主要依赖：
- `torch==2.9.0`
- `ms-swift>=4.3.1`
- `flash-attn==2.8.3`
- `lm_eval`
- `fast-hadamard-transform`

### 2. 安装 HadaQuant CUDA 扩展

HadaQuant 包提供 128 块级 Hadamard 变换与 2-bit 解量化的融合 CUDA 内核：

```bash
cd hadaquant/csrc
pip install -e .
cd ../..
```

当前编译目标包括 SM 80/86/89/90（A100、RTX 3090、RTX 4090、H100）。如需增减 GPU 架构，请修改 `setup.py`。

## 执行流程

完整流程分为四个步骤：

1. **PTQ 初始化** — `unisvq_qwen3_1.7B.sh`
   - 计算 Hessian 矩阵以指导最优量化
   - 量化权重并微调重建误差
   - 转换为 HuggingFace 格式

2. **Checkpoint 转换** — `scripts/create_qwen3_ckpt_from_quip.py`
   - 将 PTQ 权重量化结果转换为 ms_swift 可训练的 checkpoint
   - 在 128×128 块上应用 Hadamard 变换
   - 初始化线性码本参数

3. **QAT 训练** — `lc_qat_1.7B.sh`
   - 使用 ms_swift + DeepSpeed 进行分布式 QAT 微调

4. **评测**
   - LM 基准测试：`lm_eval.sh`
   - 推理吞吐测试：`quantized_infer.py`（需预先安装 `hadaquant`）

## 使用方法

### 第一步：PTQ 初始化

修改 `unisvq_qwen3_1.7B.sh` 中的路径配置：

```bash
base_model_path=YOUR_BASE_MODEL_PATH    # 如 Qwen3-1.7B 模型路径
exp_dir=YOUR_EXP_DIR                    # 输出目录
dataset_path="YOUR_DATASET_PATH/pajama.jsonl"  # 校准数据集路径
```
我们的校准数据集从`togethercomputer/RedPajama-Data-1T-Sample`中获取。  
脚本分为 3 个阶段（通过 `stage` 变量控制）：
- **阶段 1**（`stage=1`）：计算 Hessian 矩阵
- **阶段 2**（`stage=2`）：量化权重并微调
- **阶段 3**（`stage=3`）：转换为 HuggingFace 格式

```bash
bash unisvq_qwen3_1.7B.sh
```

### 第二步：Checkpoint 转换

将 PTQ checkpoint 转换为 ms_swift 可训练的格式：

```bash
python scripts/create_qwen3_ckpt_from_quip.py \
    --ori_model_path YOUR_BASE_MODEL_PATH \
    --init_ckpt_path ${exp_dir}/hf${exp_suffix} \
    --output_path ${exp_dir}/Qwen3-1.7B-pretrain/init_ckpt
```

该脚本完成以下工作：
1. 加载 PTQ 量化权重及缩放因子（SU、SV、码本）
2. 初始化线性码本的可训练权重
3. 保存为 ms_swift 兼容格式

### 第三步：QAT 训练

修改 `lc_qat_1.7B.sh` 并设置路径，然后运行：

```bash
bash lc_qat_1.7B.sh
```

使用 `ms_swift` 结合 DeepSpeed 进行分布式 QAT 微调。训练配置在 `configs/1.7B.yaml` 中指定（ms_swift 格式）。默认设置：
- 4 卡训练（`NPROC_PER_NODE=4`）
- 使用 `model/register.py` 中注册的 `qwen3_lcqat` 模型类型

### 第四步：评测

#### LM 基准评测

修改 `lm_eval.sh` 中的 `exp_dir` 和 `origin_model_path`，然后运行：

```bash
bash lm_eval.sh
```

该脚本会解包 QAT checkpoint 并在标准基准上评测：ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、PIQA、WinoGrande。

#### 推理吞吐评测

确保已安装 HadaQuant CUDA 扩展后：

```bash
./quantized_infer.sh
```

## 关键技术细节

### 线性码本（LCVQ）

传统向量量化使用离散码本查表，需要直通估计器（STE）传递梯度。我们的线性码本（[`lib/codebook/index_codebook.py`](lib/codebook/index_codebook.py)）将其替换为正交线性投影，使用固定的正交矩阵作为投影，既可以获得超过传统2bit标量方法的性能，又可以通过QAT方法训练全部量化权重。

### 块级 Hadamard 变换

原始方法在完整的输入/输出维度上执行 Hadamard 变换，推理开销较大。我们将变换限制在 **128×128 的块**上，通过CUDA 内核融合结合Hadamard变换和SU/SV缩放，同时允许通过tensor parallel方法通过分布式方法训练模型。

### 可微量化（DGE）

QAT 训练中，我们使用Ruizhe Wang等人在 Optimizing Large Language Model Training Using FP4 Quantization 中提出的[可微梯度估计器](https://arxiv.org/abs/2501.17116)。我们进一步使用随机梯度掩码以稳定训练，并通过码本投影、缩放因子（SU/SV）和隐权重的端到端训练提升性能。

## 引用

如果您在研究中使用了本代码，请引用：


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