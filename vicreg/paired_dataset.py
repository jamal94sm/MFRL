"""paired_dataset.py -- two-view STL-10 dataset for VICReg, following the
ORIGINAL VICReg two-branch augmentation recipe (facebookresearch/vicreg's
augmentations.py): asymmetric blur/solarization across the two views.

Built directly on torchvision.datasets.STL10 rather than routing through
Datasets.load_stl10, so it's independent of whatever transform that loader
bakes in -- deliberately, so VICReg's augmentation is its own paper-faithful
recipe rather than inheriting or duplicating the proposed method's. Training
SET SIZE still matches: same split (see --stl10_split), same root path
(via Datasets.data_bank_root), so both methods pretrain over the same
number of images with the same batch_size -> same steps/epoch.

Crop size = image_size[0] (96 for STL-10, not the official 224). Crop scale
(0.2, 1.0) instead of the official default (0.08, 1.0) -- an 8%-area crop
of a 96x96 image is ~27x27px, too small on this dataset; (0.2, 1.0) is the
common scale used for STL-10 self-supervised pretraining in the literature.
Normalization uses MFRL's own per-dataset stats (must match MyFuncs.py's
_DATASET_NORM so eval-time normalization is consistent).
"""

import numpy as np
from PIL import ImageOps, ImageFilter
from torchvision import transforms
from torchvision.datasets import STL10

# Mirrors MyFuncs.py's _DATASET_NORM -- keep in sync if that dict changes.
_DATASET_NORM = {
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "tiny-imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class _GaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class _Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img


def _build_view_transform(img_size, mean, std, blur_p, solarize_p):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _GaussianBlur(p=blur_p),
        _Solarization(p=solarize_p),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class TwoViewSTL10(STL10):
    """STL10 subclass returning (view1, view2, label) instead of (image,
    label). Same __len__ / underlying data as plain STL10 -- only
    __getitem__ and the transform differ."""

    def __init__(self, root, split, img_size, download=True):
        super().__init__(root=root, split=split, download=download,
                          transform=None)
        mean, std = _DATASET_NORM["stl10"]
        self.transform = _build_view_transform(img_size, mean, std,
                                                blur_p=1.0, solarize_p=0.0)
        self.transform_prime = _build_view_transform(img_size, mean, std,
                                                       blur_p=0.1, solarize_p=0.2)

    def __getitem__(self, idx):
        img, target = self.data[idx], -1
        img = self._to_pil(img)
        view1 = self.transform(img)
        view2 = self.transform_prime(img)
        return view1, view2, target

    @staticmethod
    def _to_pil(img_array):
        from PIL import Image
        # STL10 stores images as (C, H, W) uint8 arrays.
        return Image.fromarray(img_array.transpose(1, 2, 0))