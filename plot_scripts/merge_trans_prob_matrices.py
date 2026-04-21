"""
Merge transmission data from trans_prob_matrices folders.

For each NPZ file this script computes T(w) as the mean over all q values
(if available via tau_ph_wq), then merges all ranges into one sorted w,T table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\new_paper_results\16042026")
DEFAULT_OUTPUT = DEFAULT_ROOT / "merged_transmission_w_T_0-25.dat"


def compute_transmission(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency w and averaged transmission T for one NPZ file."""
    data = np.load(npz_path)

    if "w" not in data.files:
        raise ValueError(f"Missing key 'w' in {npz_path}")

    w = np.asarray(data["w"], dtype=float)

    if "tau_ph_wq" in data.files:
        tau_ph_wq = np.asarray(data["tau_ph_wq"], dtype=float)
        if tau_ph_wq.ndim != 2:
            raise ValueError(f"Expected tau_ph_wq with 2 dims in {npz_path}, got {tau_ph_wq.shape}")
        t = tau_ph_wq.mean(axis=1)
        return w, t

    if "trans_prob_matrix" in data.files:
        prob = np.asarray(data["trans_prob_matrix"])
        if prob.ndim == 4:
            # shape: (N_w, N_q, d, d) -> trace over d, mean over q
            tau_wq = np.real(np.trace(prob, axis1=2, axis2=3))
            t = tau_wq.mean(axis=1)
            return w, t
        if prob.ndim == 3:
            # shape: (N_w, d, d) -> already q-averaged
            t = np.real(np.trace(prob, axis1=1, axis2=2))
            return w, t
        raise ValueError(f"Unsupported trans_prob_matrix shape {prob.shape} in {npz_path}")

    raise ValueError(
        f"No supported transmission key in {npz_path}. Expected tau_ph_wq or trans_prob_matrix"
    )


def find_npz_files(root_dir: Path) -> list[Path]:
    """Find NPZ files in all trans_prob_matrices folders below root_dir."""
    files = []
    for folder in root_dir.rglob("trans_prob_matrices"):
        if folder.is_dir():
            files.extend(sorted(folder.glob("*.npz")))
    return sorted(files)


def merge_curves(
    w_list: list[np.ndarray],
    t_list: list[np.ndarray],
    w_min: float,
    w_max: float,
    dedup_decimals: int,
) -> np.ndarray:
    """Merge, sort and average duplicate frequencies."""
    all_w = np.concatenate(w_list)
    all_t = np.concatenate(t_list)

    mask = (all_w >= w_min) & (all_w <= w_max)
    all_w = all_w[mask]
    all_t = all_t[mask]

    rounded_w = np.round(all_w, dedup_decimals)
    order = np.argsort(rounded_w)
    rounded_w = rounded_w[order]
    all_t = all_t[order]

    unique_w, inverse = np.unique(rounded_w, return_inverse=True)
    sums = np.zeros_like(unique_w, dtype=float)
    counts = np.zeros_like(unique_w, dtype=int)

    np.add.at(sums, inverse, all_t)
    np.add.at(counts, inverse, 1)

    mean_t = sums / counts
    return np.column_stack((unique_w, mean_t))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge trans_prob_matrices into one w,T .dat file (q-averaged transmission)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder that contains range subfolders with trans_prob_matrices.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .dat file path.",
    )
    parser.add_argument("--w-min", type=float, default=0.0, help="Minimum frequency (inclusive).")
    parser.add_argument("--w-max", type=float, default=25.0, help="Maximum frequency (inclusive).")
    parser.add_argument(
        "--dedup-decimals",
        type=int,
        default=10,
        help="Decimal places for identifying duplicate frequency points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root.resolve()

    npz_files = find_npz_files(root_dir)
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in trans_prob_matrices under: {root_dir}")

    w_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []

    for npz_file in npz_files:
        w, t = compute_transmission(npz_file)
        if w.shape[0] != t.shape[0]:
            raise ValueError(
                f"Length mismatch in {npz_file}: len(w)={w.shape[0]}, len(T)={t.shape[0]}"
            )
        w_parts.append(w)
        t_parts.append(t)

    merged = merge_curves(w_parts, t_parts, args.w_min, args.w_max, args.dedup_decimals)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, merged, fmt="%.12g", header="w T", comments="")

    print(f"Found NPZ files: {len(npz_files)}")
    print(f"Merged points: {merged.shape[0]}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
