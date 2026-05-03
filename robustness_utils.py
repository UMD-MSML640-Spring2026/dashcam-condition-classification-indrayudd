from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROAD_CLASSES = ["dry", "snow", "wet"]
VISIBILITY_CLASSES = ["good", "poor"]
COMBINED_CLASSES = [
    "dry / good",
    "dry / poor",
    "snow / good",
    "snow / poor",
    "wet / good",
    "wet / poor",
]

ROAD_TO_IDX = {name: idx for idx, name in enumerate(ROAD_CLASSES)}
VISIBILITY_TO_IDX = {name: idx for idx, name in enumerate(VISIBILITY_CLASSES)}
IDX_TO_ROAD = {idx: name for name, idx in ROAD_TO_IDX.items()}
IDX_TO_VISIBILITY = {idx: name for name, idx in VISIBILITY_TO_IDX.items()}
IMAGE_SIZE = (224, 224)
PERTURBATION_BASE_MAXIMA = {
    "noise": 0.24,
    "blur": 6.0,
    "lighting_dark": 0.85,
    "lighting_bright": 0.85,
    "occlusion": 0.70,
}
PERTURBATION_100_PERCENT = {
    "noise": PERTURBATION_BASE_MAXIMA["noise"] * 1.00,
    "blur": PERTURBATION_BASE_MAXIMA["blur"] * 0.60,
    "lighting_dark": PERTURBATION_BASE_MAXIMA["lighting_dark"] * 1.00,
    "lighting_bright": PERTURBATION_BASE_MAXIMA["lighting_bright"] * 1.00,
    "occlusion": PERTURBATION_BASE_MAXIMA["occlusion"] * 1.00,
}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def candidate_dataset_roots() -> list[Path]:
    root = project_root()
    return [
        root / "rwvc_bdd100k_balanced_300",
        root / "drive" / "MyDrive" / "CV final project" / "rwvc_bdd100k_balanced_300",
    ]


def candidate_model_dirs() -> list[Path]:
    root = project_root()
    return [
        root / "drive" / "MyDrive" / "CV final project" / "models" / "efficientnet_b0_two_head",
        root / "models" / "efficientnet_b0_two_head",
    ]


def candidate_training_notebooks() -> list[Path]:
    root = project_root()
    return [
        root / "TrainTestEval.ipynb",
        root / "drive" / "MyDrive" / "CV final project" / "TrainTestEval.ipynb",
    ]


def resolve_dataset_root() -> Path:
    for path in candidate_dataset_roots():
        if path.exists():
            return path
    return candidate_dataset_roots()[0]


def resolve_model_dir() -> Path:
    for path in candidate_model_dirs():
        if path.exists():
            return path
    return candidate_model_dirs()[0]


DATASET_ROOT = resolve_dataset_root()
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
MODEL_DIR = resolve_model_dir()
BEST_MODEL_PATH = MODEL_DIR / "train_plus_aug_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class PerturbationConfig:
    noise_std: float = 0.0
    blur_radius: float = 0.0
    lighting_delta: float = 0.0
    occlusion_fraction: float = 0.0


def clamp_percent(percent: float) -> float:
    return max(0.0, min(100.0, float(percent)))


def raw_level_from_percent(kind: str, percent: float) -> float:
    if kind not in PERTURBATION_100_PERCENT:
        raise KeyError(f"Unknown perturbation kind: {kind}")
    return PERTURBATION_100_PERCENT[kind] * (clamp_percent(percent) / 100.0)


def perturbation_config_from_controls(
    noise_percent: float = 0.0,
    blur_percent: float = 0.0,
    lighting_percent: float = 0.0,
    occlusion_percent: float = 0.0,
) -> PerturbationConfig:
    lighting_percent = float(lighting_percent)
    if lighting_percent < 0:
        lighting_delta = -raw_level_from_percent("lighting_dark", abs(lighting_percent))
    else:
        lighting_delta = raw_level_from_percent("lighting_bright", lighting_percent)

    return PerturbationConfig(
        noise_std=raw_level_from_percent("noise", noise_percent),
        blur_radius=raw_level_from_percent("blur", blur_percent),
        lighting_delta=lighting_delta,
        occlusion_fraction=raw_level_from_percent("occlusion", occlusion_percent),
    )


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    corpus_name: str
    best_epoch: int | None
    valid_exact_match_acc: float
    valid_joint_score: float
    valid_loss: float
    test_exact_match_acc: float | None = None
    test_joint_score: float | None = None
    test_loss: float | None = None


def efficientnet_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class EfficientNetB0TwoHead(nn.Module):
    def __init__(self, num_road_classes: int, num_visibility_classes: int, train_backbone: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = train_backbone

        self.road_head = nn.Linear(in_features, num_road_classes)
        self.visibility_head = nn.Linear(in_features, num_visibility_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "road": self.road_head(features),
            "visibility": self.visibility_head(features),
        }


def build_model(train_backbone: bool = True) -> EfficientNetB0TwoHead:
    model = EfficientNetB0TwoHead(
        num_road_classes=len(ROAD_CLASSES),
        num_visibility_classes=len(VISIBILITY_CLASSES),
        train_backbone=train_backbone,
    )
    return model.to(DEVICE)


def load_manifest(dataset_root: Path = DATASET_ROOT) -> pd.DataFrame:
    manifest_path = Path(dataset_root) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path).copy()
    required_columns = {"split", "output_path", "road", "visibility"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    frame["path"] = frame["output_path"].map(lambda rel: Path(dataset_root) / rel)
    missing_files = frame.loc[~frame["path"].map(Path.exists), ["split", "output_path"]]
    if len(missing_files):
        raise FileNotFoundError(
            f"{len(missing_files)} manifest image paths are missing; first few:\n{missing_files.head()}"
        )

    return frame


def load_test_manifest(dataset_root: Path = DATASET_ROOT) -> pd.DataFrame:
    frame = load_manifest(dataset_root)
    test_frame = frame.loc[frame["split"] == "test"].copy().reset_index(drop=True)
    if test_frame.empty:
        raise ValueError("No test rows found in manifest")
    return test_frame


def _checkpoint_value(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if value is None:
        return default
    return float(value)


def list_best_checkpoints(model_dir: Path = MODEL_DIR) -> list[CheckpointInfo]:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    checkpoints: list[CheckpointInfo] = []
    for path in sorted(model_dir.glob("*_best.pt")):
        payload = torch.load(path, map_location="cpu")
        checkpoints.append(
            CheckpointInfo(
                path=path,
                corpus_name=str(payload.get("corpus_name", path.stem.replace("_best", ""))),
                best_epoch=payload.get("best_epoch"),
                valid_exact_match_acc=_checkpoint_value(payload, "valid_exact_match_acc", float("-inf")),
                valid_joint_score=_checkpoint_value(payload, "valid_joint_score", float("-inf")),
                valid_loss=_checkpoint_value(payload, "valid_loss", float("inf")),
            )
        )

    if not checkpoints:
        raise FileNotFoundError(f"No *_best.pt checkpoints found in {model_dir}")
    return checkpoints


def _read_train_test_summary_block(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text())
    for cell in reversed(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            text_plain = data.get("text/plain")
            if not text_plain:
                continue
            text = "".join(text_plain) if isinstance(text_plain, list) else str(text_plain)
            if "corpus_name" in text and "joint_score" in text and "exact_match_acc" in text:
                return text
    raise ValueError(f"Could not find summary dataframe output in {notebook_path}")


def load_observed_test_results(notebook_path: Path | None = None) -> dict[str, dict[str, float]]:
    candidate_paths = [Path(notebook_path)] if notebook_path is not None else candidate_training_notebooks()
    summary_text = None
    for candidate in candidate_paths:
        if candidate.exists():
            try:
                summary_text = _read_train_test_summary_block(candidate)
                break
            except ValueError:
                continue
    if summary_text is None:
        raise FileNotFoundError("Could not locate an executed TrainTestEval.ipynb with summary outputs")

    corpus_rows: dict[int, str] = {}
    metric_rows: dict[int, dict[str, float]] = {}
    known_corpuses = {"baseline_train", "train_plus_synth", "train_plus_aug", "train_plus_synth_plus_aug"}

    for raw_line in summary_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        row_idx = int(parts[0])
        if len(parts) >= 2 and parts[1] in known_corpuses:
            corpus_rows[row_idx] = parts[1]
            continue
        if len(parts) >= 4 and parts[1].startswith("/"):
            try:
                metric_rows.setdefault(row_idx, {}).update(
                    {
                        "loss": float(parts[-2]),
                        "road_loss": float(parts[-1]),
                    }
                )
            except ValueError:
                pass
        if len(parts) >= 6:
            try:
                metric_rows.setdefault(row_idx, {}).update(
                    {
                        "visibility_loss": float(parts[1]),
                        "road_acc": float(parts[2]),
                        "visibility_acc": float(parts[3]),
                        "joint_score": float(parts[4]),
                        "exact_match_acc": float(parts[5]),
                    }
                )
            except ValueError:
                pass

    results: dict[str, dict[str, float]] = {}
    for row_idx, corpus_name in corpus_rows.items():
        metrics = metric_rows.get(row_idx)
        if metrics:
            results[corpus_name] = metrics

    if not results:
        raise ValueError("Could not parse observed test results from TrainTestEval.ipynb")
    return results


def select_best_checkpoint(model_dir: Path = MODEL_DIR, notebook_path: Path | None = None) -> CheckpointInfo:
    checkpoints = list_best_checkpoints(model_dir)
    checkpoint_by_corpus = {checkpoint.corpus_name: checkpoint for checkpoint in checkpoints}

    try:
        observed_results = load_observed_test_results(notebook_path=notebook_path)
        best_corpus, best_metrics = sorted(
            observed_results.items(),
            key=lambda entry: (
                -entry[1]["exact_match_acc"],
                -entry[1]["joint_score"],
                entry[1].get("loss", float("inf")),
                entry[0],
            ),
        )[0]
        chosen = checkpoint_by_corpus[best_corpus]
        return CheckpointInfo(
            path=chosen.path,
            corpus_name=chosen.corpus_name,
            best_epoch=chosen.best_epoch,
            valid_exact_match_acc=chosen.valid_exact_match_acc,
            valid_joint_score=chosen.valid_joint_score,
            valid_loss=chosen.valid_loss,
            test_exact_match_acc=best_metrics.get("exact_match_acc"),
            test_joint_score=best_metrics.get("joint_score"),
            test_loss=best_metrics.get("loss"),
        )
    except Exception:
        return sorted(
            checkpoints,
            key=lambda item: (
                -item.valid_exact_match_acc,
                -item.valid_joint_score,
                item.valid_loss,
                item.path.name,
            ),
        )[0]


def load_checkpoint_model(checkpoint_path: Path | str) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=DEVICE)
    train_backbone = bool(payload.get("config", {}).get("train_backbone", True))
    model = build_model(train_backbone=train_backbone)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def apply_perturbations(image: Image.Image, config: PerturbationConfig) -> Image.Image:
    image = image.convert("RGB")

    if config.noise_std > 0:
        array = np.asarray(image).astype(np.float32) / 255.0
        noise = np.random.normal(loc=0.0, scale=config.noise_std, size=array.shape)
        array = np.clip(array + noise, 0.0, 1.0)
        image = Image.fromarray((array * 255.0).astype(np.uint8))

    if config.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    if config.lighting_delta != 0:
        brightness_factor = max(0.2, 1.0 + config.lighting_delta)
        contrast_factor = max(0.2, 1.0 + (config.lighting_delta * 0.5))
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    if config.occlusion_fraction > 0:
        width, height = image.size
        side_ratio = max(0.0, min(1.0, config.occlusion_fraction)) ** 0.5
        occ_width = max(1, int(width * side_ratio))
        occ_height = max(1, int(height * side_ratio))
        left = (width - occ_width) // 2
        top = (height - occ_height) // 2
        patch = Image.new("RGB", (occ_width, occ_height), color=(0, 0, 0))
        image = image.copy()
        image.paste(patch, (left, top))

    return image


class RobustnessDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, perturbation: PerturbationConfig | None = None):
        self.frame = frame.reset_index(drop=True).copy()
        self.perturbation = perturbation or PerturbationConfig()
        self.transform = efficientnet_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = apply_perturbations(image, self.perturbation)
        tensor = self.transform(image)
        labels = {
            "road": torch.tensor(ROAD_TO_IDX[row["road"]], dtype=torch.long),
            "visibility": torch.tensor(VISIBILITY_TO_IDX[row["visibility"]], dtype=torch.long),
        }
        return tensor, labels


def make_loader(frame: pd.DataFrame, perturbation: PerturbationConfig | None = None, batch_size: int = 16) -> DataLoader:
    dataset = RobustnessDataset(frame, perturbation=perturbation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


CRITERION = nn.CrossEntropyLoss()


def compute_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]):
    road_loss = CRITERION(outputs["road"], labels["road"])
    visibility_loss = CRITERION(outputs["visibility"], labels["visibility"])
    return road_loss + visibility_loss, road_loss, visibility_loss


def encode_combined_labels(road_labels: list[int], visibility_labels: list[int]) -> list[int]:
    return [road * len(VISIBILITY_CLASSES) + visibility for road, visibility in zip(road_labels, visibility_labels)]


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[dict[str, float], dict[str, list[int]]]:
    model.eval()
    totals = {
        "loss": 0.0,
        "road_loss": 0.0,
        "visibility_loss": 0.0,
        "road_correct": 0,
        "visibility_correct": 0,
        "joint_score": 0.0,
        "exact_match_correct": 0,
        "n": 0,
    }
    road_true: list[int] = []
    road_pred: list[int] = []
    visibility_true: list[int] = []
    visibility_pred: list[int] = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = {key: value.to(DEVICE) for key, value in labels.items()}

        outputs = model(images)
        loss, road_loss, visibility_loss = compute_loss(outputs, labels)

        road_predictions = outputs["road"].argmax(dim=1)
        visibility_predictions = outputs["visibility"].argmax(dim=1)
        road_matches = road_predictions == labels["road"]
        visibility_matches = visibility_predictions == labels["visibility"]
        batch_size = images.size(0)

        totals["loss"] += loss.item() * batch_size
        totals["road_loss"] += road_loss.item() * batch_size
        totals["visibility_loss"] += visibility_loss.item() * batch_size
        totals["road_correct"] += road_matches.sum().item()
        totals["visibility_correct"] += visibility_matches.sum().item()
        totals["joint_score"] += (0.5 * road_matches.float() + 0.5 * visibility_matches.float()).sum().item()
        totals["exact_match_correct"] += (road_matches & visibility_matches).sum().item()
        totals["n"] += batch_size

        road_true.extend(labels["road"].cpu().tolist())
        road_pred.extend(road_predictions.cpu().tolist())
        visibility_true.extend(labels["visibility"].cpu().tolist())
        visibility_pred.extend(visibility_predictions.cpu().tolist())

    metrics = {
        "loss": totals["loss"] / totals["n"],
        "road_loss": totals["road_loss"] / totals["n"],
        "visibility_loss": totals["visibility_loss"] / totals["n"],
        "road_acc": totals["road_correct"] / totals["n"],
        "visibility_acc": totals["visibility_correct"] / totals["n"],
        "joint_score": totals["joint_score"] / totals["n"],
        "exact_match_acc": totals["exact_match_correct"] / totals["n"],
    }
    predictions = {
        "road_true": road_true,
        "road_pred": road_pred,
        "visibility_true": visibility_true,
        "visibility_pred": visibility_pred,
        "combined_true": encode_combined_labels(road_true, visibility_true),
        "combined_pred": encode_combined_labels(road_pred, visibility_pred),
    }
    return metrics, predictions


@torch.no_grad()
def predict_single_image(
    model: nn.Module,
    image: Image.Image,
    perturbation: PerturbationConfig | None = None,
) -> dict[str, Any]:
    perturbation = perturbation or PerturbationConfig()
    processed = apply_perturbations(image, perturbation)
    tensor = efficientnet_transform()(processed).unsqueeze(0).to(DEVICE)
    outputs = model(tensor)

    road_probs = torch.softmax(outputs["road"], dim=1)[0].cpu().tolist()
    visibility_probs = torch.softmax(outputs["visibility"], dim=1)[0].cpu().tolist()
    road_idx = int(np.argmax(road_probs))
    visibility_idx = int(np.argmax(visibility_probs))
    combined_idx = road_idx * len(VISIBILITY_CLASSES) + visibility_idx

    return {
        "perturbed_image": processed,
        "road_prediction": IDX_TO_ROAD[road_idx],
        "visibility_prediction": IDX_TO_VISIBILITY[visibility_idx],
        "combined_prediction": COMBINED_CLASSES[combined_idx],
        "road_probs": dict(zip(ROAD_CLASSES, road_probs)),
        "visibility_probs": dict(zip(VISIBILITY_CLASSES, visibility_probs)),
        "road_idx": road_idx,
        "visibility_idx": visibility_idx,
    }


def test_gallery_items(test_frame: pd.DataFrame, thumbnail_size: tuple[int, int] = (160, 90)) -> list[tuple[Image.Image, str]]:
    items: list[tuple[Image.Image, str]] = []
    for idx, row in test_frame.reset_index(drop=True).iterrows():
        image = Image.open(row["path"]).convert("RGB")
        thumb = ImageOps.contain(image, thumbnail_size)
        label = f"{idx:02d} | {row['road']} / {row['visibility']}"
        items.append((thumb, label))
    return items
