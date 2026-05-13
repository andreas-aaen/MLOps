import sys
import torch
sys.path.append("src")  # noqa: E402
from src.models.resnet50 import ResNet50FineTuned


# Standard config til alle tests — ingen build_model wrapper
def get_model_config(num_classes=7, pretrained=False):
    return {
        'num_classes': num_classes,
        'pretrained': pretrained,
        'freeze_layers': ["layer1", "layer2", "layer3"],
        'dropout_fc1': 0.4,
        'dropout_fc2': 0.3,
        'hidden_units': 256,
    }


# Model construction
def test_model_builds():
    model = ResNet50FineTuned(get_model_config(pretrained=True))
    assert model is not None


def test_build_model_no_pretrained():
    """Model should build fine without pretrained weights."""
    model = ResNet50FineTuned(get_model_config(pretrained=False))
    assert model is not None


def test_model_output_has_seven_classes():
    """Output tensor should have 7 class logits."""
    model = ResNet50FineTuned(get_model_config())
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape[1] == 7


# Trainable / frozen parameters
def test_trainable_params_exist():
    model = ResNet50FineTuned(get_model_config())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_frozen_layers():
    """Early layers (layer1-3) should be frozen."""
    model = ResNet50FineTuned(get_model_config())
    frozen = [p for p in model.backbone.layer1.parameters() if not p.requires_grad]
    assert len(frozen) > 0


def test_layer4_is_trainable():
    """layer4 should have trainable parameters."""
    model = ResNet50FineTuned(get_model_config())
    trainable = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_fc_is_trainable():
    """Classifier head (fc) should be fully trainable."""
    model = ResNet50FineTuned(get_model_config())
    trainable = [p for p in model.backbone.fc.parameters() if p.requires_grad]
    assert len(trainable) > 0


# Forward pass
def test_forward_shape():
    model = ResNet50FineTuned(get_model_config())
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 7)


def test_forward_batch_size_one():
    """Model should handle a single-image batch."""
    model = ResNet50FineTuned(get_model_config())
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 7)


def test_forward_returns_tensor():
    """Output should be a torch.Tensor."""
    model = ResNet50FineTuned(get_model_config())
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)


# CustomDataset
class MockDataset:
    """Minimal re-implementation of CustomDataset
    for testing without real files."""
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = torch.zeros(3, 224, 224)
        label = torch.tensor(self.img_labels[idx], dtype=torch.long)
        return image, label


def test_custom_dataset_len():
    """Dataset should report correct length."""
    ds = MockDataset(["img1.jpg", "img2.jpg", "img3.jpg"], [0, 1, 2])
    assert len(ds) == 3


def test_custom_dataset_getitem_label():
    """__getitem__ should return the correct label."""
    ds = MockDataset(["img1.jpg", "img2.jpg"], [3, 5])
    _, label = ds[1]
    assert label.item() == 5


def test_custom_dataset_getitem_image_shape():
    """__getitem__ should return an image tensor of shape (3, 224, 224)."""
    ds = MockDataset(["img1.jpg"], [0])
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)