import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class AugmentationFactory:
    """Factory class for creating augmentation pipelines"""

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> A.Compose:
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.8, 1.0)
                ),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MedianBlur(blur_limit=(3, 7)),
                    ],
                    p=0.3,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: int = 224) -> A.Compose:
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
