import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from typing import Tuple, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model once
RESNET = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_mtcnn(keep_all: bool = True):
    return MTCNN(keep_all=keep_all, device=device)


def _transform_image(rgb_image: np.ndarray) -> torch.Tensor:
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    pil_img = Image.fromarray(rgb_image)
    return transform(pil_img).unsqueeze(0).to(device)


def get_embedding(rgb_image: np.ndarray) -> np.ndarray:
    tensor = _transform_image(rgb_image)
    with torch.no_grad():
        emb = RESNET(tensor).cpu().numpy().flatten()
    return emb.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_pca(pca_path: str):
    import pickle

    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA file not found: {pca_path}")

    with open(pca_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "pca_model" in obj:
        return obj["pca_model"]
    return obj


def load_hyperplanes(dat_path: str) -> np.ndarray:
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"NeuralHash .dat file not found: {dat_path}")

    arr = np.fromfile(dat_path, dtype=np.int8).reshape(-1, 128).astype(np.float32)
    return arr[:96]


def compute_hash(embedding_128: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    projections = np.dot(hyperplanes, embedding_128)
    return (projections > 0).astype(np.uint8)


def neuralhash_from_embedding(emb512: np.ndarray, pca_model, hyperplanes) -> np.ndarray:
    emb128 = pca_model.transform([emb512])[0]
    bits96 = compute_hash(emb128, hyperplanes)
    return bits96


def bits_to_grouped_binary(bits: np.ndarray, group_size: int = 8) -> str:
    chunks = []
    for i in range(0, len(bits), group_size):
        chunk = "".join(str(int(b)) for b in bits[i:i + group_size])
        chunks.append(chunk)
    return " ".join(chunks)


__all__ = [
    "device",
    "get_mtcnn",
    "get_embedding",
    "cosine_similarity",
    "load_pca",
    "load_hyperplanes",
    "neuralhash_from_embedding",
    "bits_to_grouped_binary",
]
