## ⚙️ Prerequisites

To set up the environment, please follow these steps:

1.  **Create a Conda environment:**
    ```bash
    conda create -n tta python=3.8.1
    conda activate tta
    conda install -y ipython pip
    ```

2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Preparation

### 💾 Datasets

To run one of the following benchmark tests, you need to download the corresponding dataset.

*   **CIFAR100 → CIFAR100-C**:
    *   The CIFAR100-C dataset is automatically downloaded when running the experiments.
    *   Alternatively, you can manually download it from [Zenodo 🔗](https://zenodo.org/record/3555552#.ZDES-XZBxhE).
*   **ImageNet → ImageNet-C**:
    *   Download the [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [Zenodo 🔗](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
*   **ImageNet → ImageNet-3DCC**:
    *   Download the [ImageNet-3DCC 🔗](https://github.com/hendrycks/robustness) dataset from the [EPFL-VILAB GitHub repository 🔗](https://github.com/EPFL-VILAB/3DCommonCorruptions?tab=readme-ov-file#3dcc-data).

> **Note:** For non-source-free methods (like RMT, etc.), you will also need to download the original [ImageNet 🔗](https://www.image-net.org/download.php) dataset.

### 🧠 Models

For the Test-Time Adaptation (TTA) benchmarks, we utilize pre-trained models from:

*   [RobustBench](https://github.com/RobustBench/robustbench)
*   [Torchvision](https://pytorch.org/vision/0.14/models.html)
*   [Timm (Hugging Face)](https://huggingface.co/timm)

---

## ▶️ Run Experiments

We provide Python scripts and Bash scripts to run the experiments.

**Using Python Scripts:**

For example, to run the `IMAGENET → IMAGNET-C` benchmark with the `RPTF` adapter, execute:

```bash
python L-CTTA.py -acfg configs/adapter/imagenet/RPTF.yaml -dcfg configs/dataset/imagenet.yaml -ocfg configs/order/imagenet/0.yaml SEED 0
```

**Using Bash Scripts:**

For example, to run experiments defined in `run.sh` and log the output:

```bash
nohup bash run.sh > run.log 2>&1 &
```
This command runs the script in the background, detached from the terminal, and redirects standard output and standard error to `run.log`.

---

## 🏆 Competitors

This repository currently supports the following Test-Time Adaptation methods:

*   [**TEA**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_TEA_Test-time_Energy_Adaptation_CVPR_2024_paper.pdf) (Test-time Energy Adaptation)
*   [**BN**](https://arxiv.org/pdf/1603.04779.pdf) (Batch Normalization Adaptation)
*   [**Tent**](https://openreview.net/pdf?id=uXl3bZLkr3c) (Test-time Entropy Minimization)
*   [**CoTTA**](https://arxiv.org/abs/2203.13591) (Continual Test-Time Adaptation)
*   [**SAR**](https://openreview.net/pdf?id=g2YraF75Tj) (Stochastic Aggregation and ReNormalization)
*   [**RoTTA**](https://openaccess.thecvf.com/content/CVPR2023/papers/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.pdf) (Robust Test-Time Adaptation in Dynamic Scenarios)
*   [**TRIBE**](https://ojs.aaai.org/index.php/AAAI/article/view/29435) (Test-Time Robustness Improvement by Balancing Efficiency and Efficacy)

---

## 🙏 Acknowledgements

This project builds upon the excellent work from several open-source projects. We extend our sincere gratitude to their authors and contributors:

*   **RobustBench**: [Official Repository 🔗](https://github.com/RobustBench/robustbench)
*   **CoTTA**: [Official Repository 🔗](https://github.com/qinenergy/cotta)
*   **Tent**: [Official Repository 🔗](https://github.com/DequanWang/tent)
*   **SAR**: [Official Repository 🔗](https://github.com/mr-eggplant/SAR)
*   **RoTTA**: [Official Repository 🔗](https://github.com/BIT-DA/RoTTA)
*   **TRIBE**: [Official Repository 🔗](https://github.com/Gorilla-Lab-SCUT/TRIBE/)
*   **TEA**: [Official Repository 🔗](https://github.com/yuanyige/tea)

---

## 📧 Contact

If you have any questions, suggestions, or bug reports regarding our work, please feel free to reach out:

*   **Primary Contact**: <a href="mailto:im@xhy.im">im@xhy.im</a>
*   Alternatively, you can open an issue in this GitHub repository.

---
