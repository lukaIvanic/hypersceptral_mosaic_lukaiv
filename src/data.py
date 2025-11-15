from __future__ import annotations

import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
logger = logging.getLogger(__name__)


import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig

TensorDict = Dict[str, torch.Tensor]
CacheEntry = Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]


def _load_mosaic(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    chw = np.transpose(arr, (2, 0, 1))  # (1, H, W)
    return torch.from_numpy(chw)


def _load_hsi_cube(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        if "cube" not in f:
            raise KeyError(f"'cube' dataset missing in {path}")
        cube = np.array(f["cube"], dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError(f"HSI cube must be 3D, got {cube.shape} in {path}")

    # Accept either (H, W, C) or (C, H, W)
    if cube.shape[0] in (31, 61, 62) and cube.shape[0] < cube.shape[-1]:
        chw = cube
    else:
        chw = np.transpose(cube, (2, 0, 1))
    return torch.from_numpy(chw)


def _resize_tensor(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "area",
) -> torch.Tensor:
    """
    Resize a CHW tensor to the given spatial size using the specified mode.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tensor.shape}")
    tensor_4d = tensor.unsqueeze(0)
    if mode in {"bilinear", "bicubic"}:
        resized = F.interpolate(tensor_4d, size=size, mode=mode, align_corners=False)
    else:
        resized = F.interpolate(tensor_4d, size=size, mode=mode)
    return resized.squeeze(0)


def random_flip(sample: TensorDict) -> TensorDict:
    """
    Apply random horizontal/vertical flips in a coordinated fashion.
    """
    if random.random() < 0.5:
        sample["input"] = torch.flip(sample["input"], dims=(2,))
        sample["target"] = torch.flip(sample["target"], dims=(2,))
    if random.random() < 0.5:
        sample["input"] = torch.flip(sample["input"], dims=(1,))
        sample["target"] = torch.flip(sample["target"], dims=(1,))
    return sample


def random_rotate_90(sample: TensorDict) -> TensorDict:
    """
    Apply a shared random 90° rotation to input and target.
    """
    k = random.randint(0, 3)
    if k:
        sample["input"] = torch.rot90(sample["input"], k, dims=(1, 2))
        if sample["target"].numel() > 0:
            sample["target"] = torch.rot90(sample["target"], k, dims=(1, 2))
    return sample


def random_resized_crop(
    sample: TensorDict,
    min_scale: float = 0.9,
    max_scale: float = 1.0,
) -> TensorDict:
    """
    Random resized crop within the existing spatial support.

    The crop is rescaled back to the original H×W to keep tensor shapes fixed.
    """
    if not (0.0 < min_scale <= max_scale <= 1.0):
        return sample

    mosaic = sample["input"]
    cube = sample["target"]
    if mosaic.dim() != 3:
        return sample

    _, h, w = mosaic.shape
    if h < 2 or w < 2:
        return sample

    scale = random.uniform(min_scale, max_scale)
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    if crop_h == h and crop_w == w:
        return sample

    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)

    mosaic_crop = mosaic[:, top : top + crop_h, left : left + crop_w]
    if cube.numel() > 0:
        cube_crop = cube[:, top : top + crop_h, left : left + crop_w]
    else:
        cube_crop = cube

    mosaic_resized = F.interpolate(
        mosaic_crop.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if cube_crop.numel() > 0:
        cube_resized = F.interpolate(
            cube_crop.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        cube_resized = cube_crop

    sample["input"] = mosaic_resized
    sample["target"] = cube_resized
    return sample


def random_intensity_jitter(
    sample: TensorDict,
    brightness: float = 0.05,
    contrast: float = 0.05,
) -> TensorDict:
    """
    Mild brightness/contrast jitter applied identically to input and target.

    Assumes inputs are in [0, 1] and clamps back to that range.
    """
    mosaic = sample["input"]
    cube = sample["target"]

    # Brightness jitter
    if brightness > 0.0:
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        mosaic = mosaic * b_factor
        if cube.numel() > 0:
            cube = cube * b_factor

    # Contrast jitter
    if contrast > 0.0:
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        mosaic_mean = mosaic.mean(dim=(1, 2), keepdim=True)
        mosaic = (mosaic - mosaic_mean) * c_factor + mosaic_mean
        if cube.numel() > 0:
            cube_mean = cube.mean(dim=(1, 2), keepdim=True)
            cube = (cube - cube_mean) * c_factor + cube_mean

    sample["input"] = torch.clamp(mosaic, 0.0, 1.0)
    if cube.numel() > 0:
        sample["target"] = torch.clamp(cube, 0.0, 1.0)
    else:
        sample["target"] = cube
    return sample


class Track1Dataset(Dataset):
    """
    Thin PyTorch dataset for Track 1.

    Each sample returns:
        {
            "input":  (1, H, W) mosaic tensor in [0, 1],
            "target": (61, H, W) hyperspectral cube tensor in [0, 1],
            "id":     sample identifier string,
        }
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[Callable[[TensorDict], TensorDict]] = None,
        augment: bool = False,
        resize_to: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        write_cache: bool = True,
        ram_cache: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.augment = augment
        self.resize_to = resize_to
        self.write_cache = write_cache
        self.ram_cache = ram_cache

        if split == "train":
            mosaic_dir = self.root / "train" / "mosaic"
            hsi_dir = self.root / "train" / "hsi_61"
        elif split in {"val", "validation"}:
            mosaic_dir = self.root / "test-public" / "mosaic"
            hsi_dir = self.root / "test-public" / "hsi_61"
        elif split == "test":
            mosaic_dir = self.root / "test_original" / "mosaic"
            hsi_dir = None  # ground truth not provided
        else:
            raise ValueError(f"Unknown split '{split}'. Use train/val/test.")

        if not mosaic_dir.exists():
            raise FileNotFoundError(f"Mosaic directory not found: {mosaic_dir}")

        self.targets_available = hsi_dir is not None and hsi_dir.exists()

        self.mosaic_paths = sorted(mosaic_dir.glob("*.npy"))
        if not self.mosaic_paths:
            raise RuntimeError(f"No mosaics found in {mosaic_dir}")

        if self.targets_available:
            self.hsi_map = {p.stem: p for p in hsi_dir.glob("*.h5")}
        else:
            self.hsi_map = {}

        self.ids = [p.stem for p in self.mosaic_paths]
        self._h5_files: Dict[str, h5py.File] = {}
        self._ram_cache: Dict[str, CacheEntry] = {}
        if cache_dir is not None and self.resize_to is not None:
            self.cache_dir = Path(cache_dir) / f"size{self.resize_to}" / self.split
        else:
            self.cache_dir = None

    def __len__(self) -> int:
        return len(self.mosaic_paths)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_files"] = {}
        state["_ram_cache"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_h5_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> TensorDict:
        mosaic_path = self.mosaic_paths[idx]
        sample_id = mosaic_path.stem

        cached_entry: Optional[CacheEntry] = None
        if self.ram_cache:
            cached_entry = self._ram_cache.get(sample_id)

        if cached_entry is not None:
            base_mosaic, base_cube, orig_shape = cached_entry
            mosaic = base_mosaic.clone()
            cube = base_cube.clone() if base_cube.numel() > 0 else base_cube
            logger.debug("ram-cache-hit: %s split=%s", sample_id, self.split)
        else:
            if self.cache_dir is not None:
                mosaic, cube, orig_shape = self._load_cached_sample(mosaic_path, sample_id)
            else:
                mosaic, cube, orig_shape = self._load_raw_sample(mosaic_path, sample_id)
                if self.resize_to is not None:
                    size = (self.resize_to, self.resize_to)
                    mosaic = _resize_tensor(mosaic, size, mode="area")
                    if cube.numel() > 0:
                        cube = _resize_tensor(cube, size, mode="area")

            if self.ram_cache and sample_id not in self._ram_cache:
                stored_mosaic = mosaic.detach().clone()
                stored_cube = cube.detach().clone() if cube.numel() > 0 else cube
                self._ram_cache[sample_id] = (stored_mosaic, stored_cube, orig_shape)
                logger.debug(
                    "ram-cache-store: %s split=%s | entries=%d",
                    sample_id,
                    self.split,
                    len(self._ram_cache),
                )

        example: TensorDict = {
            "input": mosaic,
            "target": cube,
            "id": torch.tensor(idx, dtype=torch.long),
            "id_str": sample_id,
            "orig_shape": orig_shape,
        }

        if self.augment and self.targets_available:
            example = random_flip(example)

        if self.transform is not None:
            example = self.transform(example)

        # Drop index tensor before returning
        sample = {
            "input": example["input"],
            "target": example["target"],
            "id": example["id_str"],
        }
        return sample

    def _load_hsi_cached(self, sample_id: str) -> torch.Tensor:
        if sample_id not in self._h5_files:
            self._h5_files[sample_id] = h5py.File(self.hsi_map[sample_id], "r")
        handle = self._h5_files[sample_id]
        cube = np.array(handle["cube"], dtype=np.float32, copy=True)
        if cube.ndim != 3:
            raise ValueError(f"HSI cube must be 3D, got {cube.shape}")
        if cube.shape[0] in (31, 61, 62) and cube.shape[0] < cube.shape[-1]:
            chw = cube
        else:
            chw = np.transpose(cube, (2, 0, 1))
        return torch.from_numpy(chw)

    def _cache_path(self, sample_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sample_id}.npz"

    def _load_raw_sample(self, mosaic_path: Path, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        start = time.perf_counter()
        mosaic = _load_mosaic(mosaic_path)
        orig_shape = (int(mosaic.shape[-2]), int(mosaic.shape[-1]))
        if self.targets_available:
            if sample_id not in self.hsi_map:
                raise KeyError(f"Missing HSI cube for id '{sample_id}'")
            cube = self._load_hsi_cached(sample_id)
        else:
            cube = torch.zeros(0)
        logger.debug(
            "raw-load: %s split=%s | mosaic+hsi %.1f ms",
            sample_id,
            self.split,
            (time.perf_counter() - start) * 1e3,
        )
        return mosaic, cube, orig_shape

    def _load_cached_sample(self, mosaic_path: Path, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        cache_path = self._cache_path(sample_id)
        if cache_path is not None:
            finalized = self._finalize_temp_cache(cache_path)
        else:
            finalized = False
        if cache_path is not None and (cache_path.exists() or finalized):
            start = time.perf_counter()
            data = np.load(cache_path, allow_pickle=False)
            mosaic = torch.from_numpy(data["mosaic"]).float()
            cube_arr = data["cube"]
            has_cube = bool(data["has_cube"][0]) if "has_cube" in data else cube_arr.size > 0
            if has_cube and cube_arr.size > 0:
                cube = torch.from_numpy(cube_arr).float()
            else:
                cube = torch.zeros(0)
            orig_dims = data["orig_shape"]
            orig_shape = (int(orig_dims[0]), int(orig_dims[1]))
            logger.debug(
                "cache-hit: %s split=%s | path=%s | load %.1f ms",
                sample_id,
                self.split,
                cache_path,
                (time.perf_counter() - start) * 1e3,
            )
            return mosaic, cube, orig_shape

        logger.debug("cache-miss: %s split=%s -> falling back to raw load", sample_id, self.split)
        mosaic, cube, orig_shape = self._load_raw_sample(mosaic_path, sample_id)
        if self.resize_to is not None:
            size = (self.resize_to, self.resize_to)
            mosaic = _resize_tensor(mosaic, size, mode="area")
            if cube.numel() > 0:
                cube = _resize_tensor(cube, size, mode="area")

        if cache_path is not None and self.write_cache:
            os.makedirs(cache_path.parent, exist_ok=True)
            if not cache_path.exists():
                tmp_name = cache_path.name + f".tmp{os.getpid()}-{uuid.uuid4().hex}"
                tmp_path = cache_path.with_name(tmp_name)
                np.savez(
                    tmp_path,
                    mosaic=mosaic.cpu().numpy(),
                    cube=cube.cpu().numpy() if cube.numel() > 0 else np.array([], dtype=np.float32),
                    orig_shape=np.array(orig_shape, dtype=np.int32),
                    has_cube=np.array([cube.numel() > 0], dtype=np.bool_),
                )
                try:
                    Path(tmp_path).replace(cache_path)
                except FileExistsError:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        return mosaic, cube, orig_shape

    def _finalize_temp_cache(self, cache_path: Path) -> bool:
        parent = cache_path.parent
        prefix = cache_path.name + ".tmp" 
        for candidate in parent.glob(prefix + "*"):
            try:
                Path(candidate).replace(cache_path)
                return True
            except OSError:
                continue
        return False



def create_dataloaders(
    cfg: TrainConfig,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct train/val dataloaders with shared configuration.
    """

    def _build_train_transform(cfg: TrainConfig) -> Optional[Callable[[TensorDict], TensorDict]]:
        transforms = []
        if cfg.aug_rotate90:
            transforms.append(random_rotate_90)
        if cfg.aug_resized_crop:
            transforms.append(lambda s: random_resized_crop(s, min_scale=0.9, max_scale=1.0))
        if cfg.aug_intensity_jitter:
            transforms.append(random_intensity_jitter)

        if not transforms:
            return None

        def _apply(sample: TensorDict) -> TensorDict:
            for t in transforms:
                sample = t(sample)
            return sample

        return _apply

    train_transform = _build_train_transform(cfg)

    train_ds = Track1Dataset(
        root=cfg.data_root,
        split="train",
        augment=augment_train,
        resize_to=cfg.resize_to,
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
        ram_cache=cfg.ram_cache,
        transform=train_transform,
    )
    val_ds = Track1Dataset(
        root=cfg.data_root,
        split="val",
        augment=False,
        resize_to=cfg.resize_to,
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
        ram_cache=cfg.ram_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader


__all__ = ["Track1Dataset", "create_dataloaders", "random_flip"]


