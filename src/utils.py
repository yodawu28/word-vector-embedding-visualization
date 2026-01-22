import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def to_3d_pca_with_sample(vectors: np.ndarray, fit_sample: int = 5000) -> np.ndarray:
    """
    vectors: (N x D)
    Fit PCA trên sample nhỏ để nhanh, rồi transform toàn bộ.
    """
    N, D = vectors.shape

    if D == 3:
        return vectors

    if D < 3:
        pad = np.zeros((N, 3 - D), dtype=vectors.dtype)
        return np.hstack([vectors, pad])

    pca = PCA(n_components=3, random_state=42)
    if N > fit_sample:
        idx = np.random.RandomState(42).choice(a=N, size=fit_sample, replace=False)
        pca.fit(vectors[idx])
    else:
        pca.fit(vectors)

    return pca.transform(vectors)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    
    return float(np.dot(a, b) / (norm_a * norm_b))

def cosine_similarity_matrix(vectors: np.ndarray, labels: list[str]) -> pd.DataFrame:
    # vectors: (N x D)
    N = vectors.shape[0]
    sims = np.zeros((N, N), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1) + 1e-9
    vn = vectors / norms[:, None]
    sims = vn @ vn.T
    return pd.DataFrame(sims, index=labels, columns=labels)


def cosine_similarity_matrix_allow_zeros(vectors: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """
    Cosine similarity cho phép vector 0:
    - Nếu vector i hoặc j có norm = 0 => similarity = NaN
    """
    V = vectors.astype(np.float32)
    norms = np.linalg.norm(V, axis=1)

    # normalize an toàn: chỗ norm=0 sẽ để 0 (không dùng)
    Vn = np.zeros_like(V)
    nz = norms > 0
    Vn[nz] = V[nz] / norms[nz, None]

    sims = Vn @ Vn.T  # (N x N)

    # set NaN cho hàng/cột có norm=0
    zero_idx = np.where(~nz)[0]
    for i in zero_idx:
        sims[i, :] = np.nan
        sims[:, i] = np.nan

    return pd.DataFrame(sims, index=labels, columns=labels)


def parse_word_list(raw: str):
    """
    Nhập nhiều từ:
    - mỗi dòng 1 từ/cụm từ
    - hoặc dùng dấu phẩy để ngăn cách
    """
    if not raw:
        return []
    # split theo newline trước, rồi split tiếp theo comma
    items = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        items.extend([p for p in parts if p])
    # normalize: lowercase + remove duplicates but keep order
    seen = set()
    out = []
    for w in items:
        w_norm = w.strip().lower()
        if w_norm and w_norm not in seen:
            seen.add(w_norm)
            out.append(w_norm)
    return out
