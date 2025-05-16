import pytest
import torch
from PIL import Image, ImageDraw
import math

# Assuming src is in PYTHONPATH or tests are run from root
from src.UoB.utils.transforms import PadToSquareAndAlign

@pytest.fixture
def target_size():
    return 224

@pytest.fixture
def fill_val():
    return 0

def test_pad_pil_grayscale(target_size, fill_val):
    """Test padding a grayscale PIL image."""
    h, w = 150, 100
    pil_img_gray = Image.new('L', (w, h), color=128)
    draw = ImageDraw.Draw(pil_img_gray)
    draw.rectangle([(10, 10), (w-10, h-10)], fill=255)

    pad_transform = PadToSquareAndAlign(size=target_size, fill=fill_val)
    padded_tensor = pad_transform(pil_img_gray)

    assert padded_tensor.shape == (1, target_size, target_size)

    expected_pad_left = (target_size - w) // 2
    expected_pad_right = target_size - w - expected_pad_left
    expected_pad_top = 0
    expected_pad_bottom = target_size - h

    # Check image area
    assert torch.any(padded_tensor[:, 0:h, expected_pad_left:(target_size - expected_pad_right)] != fill_val)
    # Check padded areas
    if expected_pad_left > 0:
        assert torch.all(padded_tensor[:, :, 0:expected_pad_left] == fill_val)
    if expected_pad_right > 0:
        assert torch.all(padded_tensor[:, :, -expected_pad_right:] == fill_val)
    if expected_pad_bottom > 0:
         assert torch.all(padded_tensor[:, -expected_pad_bottom:, :] == fill_val)
    # Check boundaries
    assert torch.any(padded_tensor[:, expected_pad_top, expected_pad_left] != fill_val)
    if expected_pad_bottom > 0:
        assert torch.all(padded_tensor[:, h, expected_pad_left] == fill_val)

def test_pad_tensor_rgb(target_size, fill_val):
    """Test padding an RGB tensor image."""
    h, w = 200, 180
    tensor_img_rgb = torch.rand(3, h, w)

    pad_transform = PadToSquareAndAlign(size=target_size, fill=fill_val)
    padded_tensor_rgb = pad_transform(tensor_img_rgb)

    assert padded_tensor_rgb.shape == (3, target_size, target_size)

    expected_pad_left_rgb = (target_size - w) // 2
    expected_pad_right_rgb = target_size - w - expected_pad_left_rgb
    expected_pad_bottom_rgb = target_size - h

    assert torch.any(padded_tensor_rgb[:, 0:h, expected_pad_left_rgb:(target_size - expected_pad_right_rgb)] != fill_val)
    if expected_pad_left_rgb > 0:
        assert torch.all(padded_tensor_rgb[:, :, 0:expected_pad_left_rgb] == fill_val)
    if expected_pad_right_rgb > 0:
        assert torch.all(padded_tensor_rgb[:, :, -expected_pad_right_rgb:] == fill_val)
    if expected_pad_bottom_rgb > 0:
        assert torch.all(padded_tensor_rgb[:, -expected_pad_bottom_rgb:, :] == fill_val)

def test_pad_already_square(target_size, fill_val):
    """Test padding an already square tensor."""
    tensor_square = torch.rand(3, target_size, target_size)
    pad_transform = PadToSquareAndAlign(size=target_size, fill=fill_val)
    padded_square = pad_transform(tensor_square)

    assert padded_square.shape == (3, target_size, target_size)
    assert torch.equal(tensor_square, padded_square)

def test_pad_larger_image(target_size, fill_val):
    """Test padding an image larger than the target size (triggers resize)."""
    h, w = 300, 250
    large_tensor = torch.rand(1, h, w)
    pad_transform = PadToSquareAndAlign(size=target_size, fill=fill_val)
    padded_large = pad_transform(large_tensor)

    assert padded_large.shape == (1, target_size, target_size)
    padding_applied = torch.any(padded_large == fill_val)
    assert padding_applied
    if padding_applied:
         assert padded_large[0, -1, 0] == fill_val
         assert padded_large[0, -1, -1] == fill_val 