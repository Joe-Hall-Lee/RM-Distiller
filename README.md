# RM-Distiller: Exploiting Generative LLM for Reward Model Distillation

This is the official repository for paper _RM-Distiller: Exploiting Generative LLM for Reward Model Distillation_.

In this paper, we introduce RM-Distiller, a framework designed to distill discriminative reward models (RMs) from generative LLMs.

---

## 📂 Repository Structure

```text
.
├── refine/            # Contrastive Refinement inluding response diagnosis and minimal editing
├── score/             # Preference strength annotation and Self-Calibrated Scoring
├── RewardTrainer/           # RM training and evaluation
│   ├── configs/             # Generative RM evaluation configuration files
│   ├── eval/                # Evaluation scripts and configuration files
│   ├── scripts/             # Entry scripts for training and evaluation
│   │   ├── train_rm.sh              # BT Classfier training
│   │   ├── train_distilrm.sh        # RM-Distiller training
│   │   ├── eval_rm.sh               # Discriminative RM evaluation
│   │   ├── eval_judge.sh            # Generative RM evaluation
│   └── train/               # Core training logic and model implementations
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── requirements.txt  # Python dependencies
```

## ⚡️ Usage

### Preparation

Please refer to the following commands to prepare your environment.

```shell
conda create -n rm-distiller python=3.12
pip install -r requirements.txt
```

### Contrastive Refinement

To synthesize highly contrastive preference pairs via teacher-guided minimal refinement, run the refinement pipeline in the `refine/` directory.

```bash
python refine/refine_response_vllm.py
```

### Self-Calibrated Scoring

To assign calibrated preference scores and obtain preference strength margins, run the scoring scripts in the `score/` directory.

```bash
python score/cali_score_vllm.py
```

### RM Training

To train the RM with Margin-Aware Regression and Generative Regularization, use the training scripts provided in `RewardTrainer/scripts/`.

```bash
cd RewardTrainer
bash scripts/train_distilrm.sh
```
