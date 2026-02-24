import sys
import torch

sys.path.append("src")
from training.model import build_model



def test_model_builds():
    model = build_model(num_classes=7, pretrained=False)
    assert model is not None


#


def test_trainable_params_exist():
    model = build_model(num_classes=7, pretrained=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_forward_shape():
    model = build_model(num_classes=7, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 7)
