---
datasets:
- emotion-recognition-dataset
language: en
library_name: pytorch
license: mit
metrics:
- accuracy
model_name: resnet50
---

## Model describtion 
The model is a **resnet50**, pretrained on ImageNet and fine-tuned for facial emotion recognition 

### architecture Detaljer:
- **Prætrænet:** True
- **Hidden Units:** 256
- **Dropout:** 0.4 (FC1) og 0.3 (FC2)
- **Frosne lag:** layer1, layer2, layer3

## Træningsparametre
- **Optimizer:** AdamW (lr=0.0001)
- **Scheduler:** OneCycleLR
- **Batch Size:** 32
- **Epochs:** 15

## Evalueringstærskel
Modellen kræver en minimum **Accuracy på 70.0%** for at blive promoveret til Production.
