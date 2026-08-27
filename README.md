# HermNet-AMC

**Harnessing Second-Order Statistics for Cross-Scenario Over-the-Air Modulation Classification**

Official PyTorch implementation of **HermNet**, a lightweight Hermitian-covariance-driven network for cross-scenario automatic modulation classification (AMC). HermNet learns modulation-discriminative representations that stay robust across **unseen** over-the-air (OTA) scenarios — without any access to target-domain data.

> On a self-collected four-scenario OTA dataset under a leave-one-domain-out protocol, HermNet reaches **89.47% average cross-scenario accuracy** with only **14.23 K trainable parameters**, outperforming eight representative baselines.

---

## Highlights

- **Hermitian Covariance Attention (HCA).** Recalibrates deep complex features using the Hermitian covariance of learned feature channels. The covariance is **invariant to a global phase rotation**, so it suppresses scenario-induced phase uncertainty (unknown initial phase, oscillator phase noise) while preserving modulation-relevant inter-channel structure. The matrix square root is computed via differentiable **Newton–Schulz iterations**, so the whole block is end-to-end trainable.
- **Amplitude–Phase Fusion.** Decomposes the recalibrated complex features into magnitude and phase and fuses them into real-valued temporal features.
- **Multi-Scale Gated Temporal Modeling.** Stacked blocks combine depthwise multi-scale convolutions `{3, 5, 7}` with a gated linear unit (GLU) inside a residual connection.
- **Lightweight.** ~14.23 K parameters, 2.81 MFLOPs — suitable for resource-constrained deployment.

---

## Repository Structure

```
.
├── main.py                          # Entry point: training / testing
├── train.sh                         # Leave-one-domain-out sweep (4 folds, 4 GPUs)
└── utils/
    ├── train_tools.py               # Data loading, train / eval / test loops
    └── model_factory/
        └── HermNet.py               # HermNet model (HCA, AP-fusion, temporal blocks)
```

> The training data is expected under `./dataset/CD2025/` and model weights are written to `./weight/`. Create these folders (or let the scripts create `logs/`) as needed.

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- numpy
- scipy
- h5py
- torchsummary

This version of the code has been tested on **PyTorch 2.5.1** with an **NVIDIA RTX 3090**.

```bash
git clone https://github.com/BeechburgPieStar/HermNet-AMC.git
cd HermNet-AMC
pip install torch==2.5.1 numpy scipy h5py torchsummary
```

---

## Dataset

The **CD2025** OTA dataset is collected with an **ADALM-Pluto SDR** frontend under four wireless channel conditions of increasing propagation complexity:

| Index | Scenario | Description |
|:-----:|:---------|:------------|
| `L1` (S1) | Outdoor open area | Line-of-sight (LoS) |
| `L2` (S2) | Indoor classroom  | LoS + moderate multipath |
| `L3` (S3) | Indoor office     | Obstructed, reflection / diffraction |
| `L4` (S4) | Outdoor corridor & staircase | Severe blockage, strong multipath & fading (hardest) |

- **10 modulations:** BPSK, QPSK, 8PSK, 2FSK, 4FSK, 8FSK, 16QAM, 64QAM, AM, FM
- Symbol rate **25 kHz**, carrier **2 GHz**, distances **1 m** and **5 m**
- **500** samples per (modulation, distance) pair → **10,000** samples per scenario → **40,000** total
- Each sample: **1024 complex baseband points**, stored as I/Q of shape `[N, 2, 1024]`

### Download

The **CD2025** dataset is hosted on Baidu Netdisk:

> **Baidu Netdisk:** <https://pan.baidu.com/s/1asKM4AI7Mf6ExAQl7q2UpQ>
> **Access code:** `3uux`

After downloading, extract the archive so that the four `.h5` files are placed directly under `./dataset/CD2025/`, keeping the file names unchanged. The expected directory structure is:

```
dataset/
└── CD2025/
    ├── L1.h5
    ├── L2.h5
    ├── L3.h5
    └── L4.h5
```

Each `.h5` file is an HDF5 container with two keys: `X` (signals, shape `[N, 2, 1024]`) and `Y` (integer labels). Signals are power-normalized per sample at load time (`power_normalization_new`).

---

## Usage

### Single leave-one-domain-out run

Train on three scenarios and test on the held-out one (here `L1,L2,L3` → `L4`):

```bash
python3 main.py \
    --mode train_test \
    --model_name HermNet \
    --dataset_name CD2025 --num_cls 10 \
    --dataset_indexes_for_train L1 L2 L3 \
    --dataset_indexes_for_test  L4 \
    --d_model 16 --fused_dim 8 --gmlp_layers 3 --reduction 2 \
    --cuda_device 0
```

### Full four-fold sweep

`train.sh` launches all four leave-one-domain-out folds in parallel (one per GPU):

```bash
bash train.sh
```

### Test only

```bash
python3 main.py \
    --mode only_test \
    --dataset_indexes_for_train L1 L2 L3 \
    --dataset_indexes_for_test  L4
```

### Key arguments

| Argument | Default | Description |
|:---------|:-------:|:------------|
| `--mode` | `train_test` | `train_test`, `only_train`, or `only_test` |
| `--dataset_indexes_for_train` | `L1 L2 L3` | Source scenarios |
| `--dataset_indexes_for_test` | `L4` | Held-out target scenario |
| `--d_model` | `16` | Complex feature dimension `C` |
| `--fused_dim` | `8` | Fused temporal dimension `D` |
| `--gmlp_layers` | `3` | Number of temporal blocks `K` |
| `--reduction` | `2` | HCA reduction ratio `η` |
| `--dropout` | `0.2` | Dropout rate |
| `--batch_size` | `512` | Global batch (split across source domains) |
| `--epochs` | `400` | Max epochs (early stopping, patience 10) |
| `--lr` | `0.001` | Adam learning rate |
| `--seed` | `2023` | Random seed |

Weights are saved to
`weight/{model_name}_dm{d_model}_fd{fused_dim}_gmlp{gmlp_layers}_red{reduction}_{version}_{test_index}.pth`.

---

## Results

Leave-one-domain-out cross-scenario accuracy and complexity (10 classes):

| Model | S1 | S2 | S3 | S4 | **Avg.** | Params (K) | MFLOPs |
|:------|:--:|:--:|:--:|:--:|:--------:|:----------:|:------:|
| MCNet     | 85.83 | 85.48 | 88.26 | 57.30 | 79.22 | 75.34 | 10.72 |
| CNN-LSTM  | 85.75 | 89.59 | 87.35 | 50.31 | 78.25 | 4282.71 | 4.20 |
| MCLDNN    | 88.68 | 89.43 | 85.48 | 66.18 | 82.44 | 370.46 | 88.31 |
| PET-CGDNN | 85.49 | 86.28 | 86.56 | 69.41 | 81.94 | 126.62 | 11.24 |
| MCformer  | 92.40 | 89.02 | 85.95 | 71.38 | 84.69 | 74.62 | 289.45 |
| FEA-T     | 92.19 | 90.44 | 87.35 | 75.93 | 86.48 | 1688.60 | 1.57 |
| TLDNN     | 92.35 | 89.09 | 85.70 | 74.46 | 85.40 | 280.76 | 15.11 |
| AWN       | 88.97 | 91.36 | 87.46 | 60.50 | 82.07 | 342.17 | 25.75 |
| **HermNet (Ours)** | **93.01** | **92.74** | **91.11** | **81.02** | **89.47** | **14.23** | 2.81 |

HermNet improves the average accuracy by **+2.99%** over the strongest baseline (FEA-T), with the gain rising to **+5.09%** on the hardest scenario S4, while using **99.16% fewer parameters**.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{mao2026hermnet,
  author  = {Mao, Jiangyuan and Wang, Yu},
  title   = {Harnessing Second-Order Statistics for Cross-Scenario
             Over-the-Air Modulation Classification},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {submitted}
}
```

---

## Acknowledgements

We sincerely thank the authors of the following works, whose open-source
implementations were used as baselines in our experiments and greatly
supported this research:

- **MCNet** — T. Huynh-The, C.-H. Hua, Q.-V. Pham, and D.-S. Kim, "MCNet: An efficient CNN architecture for robust automatic modulation classification," *IEEE Commun. Lett.*, vol. 24, no. 4, pp. 811–815, Apr. 2020.
  Code: <https://github.com/ThienHuynhThe/MCNet>
- **CNN-LSTM** — Z. Zhang, H. Luo, C. Wang, C. Gan, and Y. Xiang, "Automatic modulation classification using CNN-LSTM based dual-stream structure," *IEEE Trans. Veh. Technol.*, vol. 69, no. 11, pp. 13521–13531, Nov. 2020.
  (No official public repository; commonly reproduced via the AMR-Benchmark below.)
- **MCLDNN** — J. Xu, C. Luo, G. Parr, and Y. Luo, "A spatiotemporal multi-channel learning framework for automatic modulation recognition," *IEEE Wireless Commun. Lett.*, vol. 9, no. 10, pp. 1629–1632, Oct. 2020.
  Code: <https://github.com/wzjialang/MCLDNN>
- **PET-CGDNN** — F. Zhang, C. Luo, J. Xu, and Y. Luo, "An efficient deep learning model for automatic modulation recognition based on parameter estimation and transformation," *IEEE Commun. Lett.*, vol. 25, no. 10, pp. 3287–3290, Oct. 2021.
  Code: <https://github.com/Richardzhangxx/PET-CGDNN>
- **MCformer** — S. Hamidi-Rad and S. Jain, "MCformer: A transformer based deep neural network for automatic modulation classification," in *Proc. IEEE Global Commun. Conf. (GLOBECOM)*, 2021, pp. 1–6.
  Code: <https://github.com/InterDigitalInc/Fireball> (see `Playgrounds/MCformer`)
- **FEA-T** — Y. Chen, B. Dong, C. Liu, W. Xiong, and S. Li, "Abandon locality: Frame-wise embedding aided transformer for automatic modulation recognition," *IEEE Commun. Lett.*, vol. 27, no. 1, pp. 327–331, Jan. 2023.
  Code: <https://github.com/YTao-Chen/FEA-T>
- **TLDNN** — Y. Qu, Z. Lu, R. Zeng, J. Wang, and J. Wang, "Enhancing automatic modulation recognition through robust global feature extraction," *IEEE Trans. Cogn. Commun. Netw.*, 2025.
  Code: <https://github.com/AMR-Master/TLDNN>
- **AWN** — J. Zhang, T. Wang, Z. Feng, and S. Yang, "Toward the automatic modulation classification with adaptive wavelet network," *IEEE Trans. Cogn. Commun. Netw.*, vol. 9, no. 3, pp. 549–563, 2023.
  Code: <https://github.com/zjwfufu/AWN>

We also thank the maintainers of the **AMR-Benchmark**, a unified implementation of several baseline deep-learning models for automatic modulation recognition: <https://github.com/Richardzhangxx/AMR-Benchmark>

---

## License

This code is distributed under an MIT License. Note that our code depends on other libraries and datasets, which each have their own respective licenses that must also be followed.
