"""
Image transformation utilities.

Handles image resizing, normalization, and multi-camera processing.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import io


class ImageProcessor:
    """
    Processes images for VLA models.

    Handles resizing, normalization, and format conversion.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image processor.

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean for normalization (ImageNet defaults)
            std: Std for normalization (ImageNet defaults)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def process(
        self,
        image: Union[np.ndarray, bytes, Image.Image],
        resize: bool = True
    ) -> np.ndarray:
        """
        Process a single image.

        Args:
            image: Input image as numpy array, bytes, or PIL Image
            resize: Whether to resize the image

        Returns:
            Processed image as numpy array of shape (H, W, 3)
        """
        # Convert to PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if requested
        if resize and image.size != (self.target_size[1], self.target_size[0]):
            image = image.resize(
                (self.target_size[1], self.target_size[0]),
                Image.Resampling.BILINEAR
            )

        # Convert to numpy array
        image_np = np.array(image).astype(np.float32)

        # Normalize if requested
        if self.normalize:
            image_np = image_np / 255.0
            image_np = (image_np - self.mean) / self.std

        return image_np

    def process_batch(
        self,
        images: List[Union[np.ndarray, bytes, Image.Image]],
        resize: bool = True
    ) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            images: List of images
            resize: Whether to resize images

        Returns:
            Processed images as numpy array of shape (B, H, W, 3)
        """
        processed = [self.process(img, resize=resize) for img in images]
        return np.stack(processed, axis=0)

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize an image back to [0, 255] uint8 range.

        Args:
            image: Normalized image of shape (..., H, W, 3)

        Returns:
            Denormalized image
        """
        if self.normalize:
            image = image * self.std + self.mean
            image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


class MultiCameraProcessor:
    """
    Processes multiple camera views.

    Handles camera synchronization, padding for missing cameras, and
    multi-view image processing.
    """

    def __init__(
        self,
        camera_names: List[str],
        target_size: Tuple[int, int] = (224, 224),
        pad_missing: bool = True
    ):
        """
        Initialize multi-camera processor.

        Args:
            camera_names: List of camera names (e.g., ['base_0_rgb', 'left_wrist_0_rgb'])
            target_size: Target image size for all cameras
            pad_missing: If True, pad missing cameras with zeros
        """
        self.camera_names = camera_names
        self.target_size = target_size
        self.pad_missing = pad_missing
        self.processor = ImageProcessor(target_size=target_size)

    def process(
        self,
        images: Dict[str, Union[np.ndarray, bytes, Image.Image]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
        """
        Process multiple camera views.

        Args:
            images: Dictionary mapping camera names to images

        Returns:
            Tuple of:
                - Processed images dictionary
                - Camera mask dictionary (True if camera present, False if padded)
        """
        processed_images = {}
        camera_masks = {}

        for cam_name in self.camera_names:
            if cam_name in images and images[cam_name] is not None:
                # Process the image
                processed_images[cam_name] = self.processor.process(images[cam_name])
                camera_masks[cam_name] = True
            else:
                if self.pad_missing:
                    # Create zero-padded image
                    processed_images[cam_name] = np.zeros(
                        (*self.target_size, 3),
                        dtype=np.float32
                    )
                    camera_masks[cam_name] = False
                else:
                    raise ValueError(f"Missing camera view: {cam_name}")

        return processed_images, camera_masks

    def process_batch(
        self,
        image_batches: List[Dict[str, Union[np.ndarray, bytes, Image.Image]]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process a batch of multi-camera observations.

        Args:
            image_batches: List of dictionaries mapping camera names to images

        Returns:
            Tuple of:
                - Processed images dictionary mapping camera names to (B, H, W, 3) arrays
                - Camera mask dictionary mapping camera names to (B,) boolean arrays
        """
        all_processed = {cam: [] for cam in self.camera_names}
        all_masks = {cam: [] for cam in self.camera_names}

        for images in image_batches:
            processed, masks = self.process(images)
            for cam_name in self.camera_names:
                all_processed[cam_name].append(processed[cam_name])
                all_masks[cam_name].append(masks[cam_name])

        # Stack into batch arrays
        batch_processed = {
            cam: np.stack(all_processed[cam], axis=0)
            for cam in self.camera_names
        }
        batch_masks = {
            cam: np.array(all_masks[cam], dtype=bool)
            for cam in self.camera_names
        }

        return batch_processed, batch_masks


class DepthProcessor:
    """
    Processes depth images.

    Handles depth normalization and format conversion.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        depth_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True
    ):
        """
        Initialize depth processor.

        Args:
            target_size: Target depth map size (height, width)
            depth_range: (min_depth, max_depth) for normalization. If None, auto-compute.
            normalize: Whether to normalize depth values
        """
        self.target_size = target_size
        self.depth_range = depth_range
        self.normalize = normalize

    def process(
        self,
        depth: Union[np.ndarray, bytes],
        resize: bool = True
    ) -> np.ndarray:
        """
        Process a single depth image.

        Args:
            depth: Input depth as numpy array or bytes
            resize: Whether to resize the depth map

        Returns:
            Processed depth map of shape (H, W)
        """
        # Decode if bytes
        if isinstance(depth, bytes):
            depth = np.frombuffer(depth, dtype=np.float32)
            # Need to know the original shape to reshape properly
            # This is a limitation - ideally shape info should be stored

        # Ensure 2D
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        # Convert to PIL Image for resizing
        if resize and depth.shape != self.target_size:
            depth_img = Image.fromarray(depth.astype(np.float32))
            depth_img = depth_img.resize(
                (self.target_size[1], self.target_size[0]),
                Image.Resampling.NEAREST
            )
            depth = np.array(depth_img)

        # Normalize if requested
        if self.normalize:
            if self.depth_range is not None:
                min_depth, max_depth = self.depth_range
            else:
                min_depth = np.min(depth)
                max_depth = np.max(depth)

            # Avoid division by zero
            if max_depth - min_depth < 1e-6:
                depth = np.zeros_like(depth)
            else:
                depth = (depth - min_depth) / (max_depth - min_depth)

        return depth.astype(np.float32)
