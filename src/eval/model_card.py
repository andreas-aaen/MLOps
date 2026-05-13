import yaml
import os
from huggingface_hub import ModelCard, ModelCardData

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# 1. Indlæs konfigurationer
train_cfg = load_yaml('src/config/train_config.yaml')
test_cfg = load_yaml('src/config/test_config.yaml')

# 2. Opsæt Metadata
card_data = ModelCardData(
    language='en',
    license='mit',
    library_name='pytorch',
    model_name=train_cfg['model']['name'],
    datasets=['emotion-recognition-dataset'], 
    metrics=['accuracy']
)

# 3. Skræddersyet indhold
content = f"""
# Emotion Classifier ({train_cfg['model']['name']})

A fine-tuned vision model for emotion detection in images.

## Model Beskrivelse
Modellen er en **{train_cfg['model']['name']}**, prætrænet på ImageNet og fine-tuned til følelsesgenkendelse.

### Arkitektur Detaljer:
- **Prætrænet:** {train_cfg['model'].get('pretrained', True)}
- **Hidden Units:** {train_cfg['model'].get('hidden_units', 256)}
- **Dropout:** {train_cfg['model'].get('dropout_fc1', 0.4)} (FC1) og {train_cfg['model'].get('dropout_fc2', 0.3)} (FC2)
- **Frosne lag:** {", ".join(train_cfg['model'].get('freeze_layers', []))}

## Træningsparametre
- **Optimizer:** {train_cfg['optimizer']['name']} (lr={train_cfg['optimizer']['lr']})
- **Scheduler:** {train_cfg['scheduler']['name']}
- **Batch Size:** {train_cfg['train']['batch_size']}
- **Epochs:** {train_cfg['train']['epochs']}

## Evaluering og Anvendelse
- **Evalueringstærskel:** Modellen kræver en minimum **Accuracy på {test_cfg.get('accuracy_threshold', 0.7) * 100}%** for at blive promoveret til Production.
- **Direct Use:** Realtids emotion-analyse i kontrollerede miljøer.
- **Begrænsninger:** Bør ikke bruges til automatiseret ansættelse eller klinisk diagnose. Modellen kan have bias baseret på demografien i træningsdataet.

## How to Get Started
```python
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/resnet50-emotion-classifier/Staging")