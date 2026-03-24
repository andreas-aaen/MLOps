# Model Card — Facial Emotion Recognition (ResNet-50)

**Developers:** Lanja Mozafar Khorshid, Mikkel Stouby Holm & Andreas Aaen
**Model type:** Fine-tuned ResNet-50, 7-class image classifier

## Code
**Model & training script:** https://github.com/Lanja11/deep-learning-mini-project
**Commit hash:** 928df36

**MLOps pipeline:** https://github.com/andreas-aaen/MLOps
**Commit hash:** 8d6a25a *(updated per training run)*

## Data
**Dataset:** FER2013 + RAF-DB combined (~49,000 images, 7 classes)
**Version:** Tracked via DVC - see `Dataset.dvc` in MLOps repository
**Remote storage:** MinIO S3 bucket `lam`, AAU MLOps cluster (172.24.198.42:9001)

## Experiments / Runs
**Experiment tracking:** MLflow - AAU MLOps cluster (port 5050)
**Run ID:** *(logged per training run - see MLflow registry)*

## Infrastructure
**Training environment:** AAU AI-Lab
**CI/CD:** Jenkins - AAU MLOps cluster (port 8080)
**Configuration:** `train_config.yaml` in MLOps repository
**Saved model:** `best_resnet50_emotion1.pth`

## Evaluation Results
**Accuracy:** 0.74 - **Macro F1:** 0.72

| Class    | F1   |
|----------|------|
| angry    | 0.62 |
| disgust  | 0.89 |
| fear     | 0.55 |
| happy    | 0.89 |
| neutral  | 0.68 |
| sad      | 0.61 |
| surprise | 0.80 |

## Risks, Biases and Known Issues
- Performs poorly on visually similar emotions (fear, sad, angry)
- No disaggregated evaluation across demographic groups has been performed
- Not suitable for high-stakes decisions (hiring, law enforcement, clinical use)

---

*Last updated: March 2026 — to be completed as the course progresses.*
