from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_load_curve import format_hour_axis
from eda_peer_comparison import collect_customer_profiles
from reader import AusgridReader


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEASON_COLORS = ["#24708F", "#C4592E", "#2F7657", "#9A7611", "#6B4D7B", "#A44542"]


def _euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )


def _kmeans_pp_init(
    values: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = [values[rng.integers(0, len(values))]]
    distances = np.full(len(values), np.inf)
    for _ in range(1, clusters):
        latest = centers[-1]
        distances = np.minimum(
            distances,
            ((values - latest) ** 2).sum(axis=1),
        )
        probability = distances / distances.sum()
        centers.append(values[rng.choice(len(values), p=probability)])
    return np.vstack(centers)


def _kmeans_once(
    values: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
    *,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = _kmeans_pp_init(values, clusters, rng)
    labels = np.full(len(values), -1, dtype=int)
    for _ in range(max_iter):
        distances = _euclidean_distances(values, centers)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                farthest = distances.min(axis=1).argmax()
                centers[cluster] = values[farthest]
    distances = _euclidean_distances(values, centers)
    inertia = float((distances.min(axis=1) ** 2).sum())
    return labels, centers, inertia


def kmeans_numpy(
    values: np.ndarray,
    clusters: int,
    *,
    n_init: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        result = _kmeans_once(values, clusters, rng)
        if best is None or result[2] < best[2]:
            best = result
    assert best is not None
    return best


def silhouette_score_numpy(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    distances = _euclidean_distances(values, values)
    scores: list[float] = []
    for index in range(len(values)):
        own = labels == labels[index]
        own[index] = False
        if own.sum() == 0:
            scores.append(0.0)
            continue
        a = distances[index, own].mean()
        b = min(
            distances[index, labels == cluster].mean()
            for cluster in np.unique(labels)
            if cluster != labels[index]
        )
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def _label_clusters(centroids: np.ndarray) -> dict[int, str]:
    labels: dict[int, str] = {}
    for cluster, centroid in enumerate(centroids):
        normalized = centroid / centroid.mean()
        peak_slot = int(normalized.argmax())
        peak_hour = peak_slot / 2
        variation = float(normalized.std())

        if variation < 0.22:
            base = "平稳型"
        elif 17 <= peak_hour < 21:
            evening_ratio = float(
                normalized[34:42].mean() / normalized.mean()
            )
            if evening_ratio >= 1.6:
                base = "强晚高峰型"
            elif evening_ratio >= 1.25:
                base = "缓晚高峰型"
            else:
                base = "晚高峰型"
        elif peak_hour >= 21:
            base = "夜间型"
        elif 4 <= peak_hour < 11:
            morning_ratio = float(
                normalized[10:18].mean() / normalized.mean()
            )
            if morning_ratio >= 1.6:
                base = "强早高峰型"
            elif morning_ratio >= 1.25:
                base = "缓早高峰型"
            else:
                base = "早高峰型"
        else:
            base = "日间多峰型"

        final = base
        suffix = 2
        while final in labels.values():
            final = f"{base}{suffix}"
            suffix += 1
        labels[cluster] = final
    return labels


def cluster_profile_matrix(
    matrix: pd.DataFrame,
    *,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    values = matrix.to_numpy(dtype=float)
    normalized = values / values.mean(axis=1, keepdims=True)

    score_rows: list[dict[str, float | int]] = []
    best: tuple[int, np.ndarray, np.ndarray, float] | None = None
    min_cluster_size = max(5, int(len(matrix) * 0.02))
    for clusters in range(k_min, k_max + 1):
        labels, centers, inertia = kmeans_numpy(
            normalized,
            clusters,
            random_state=random_state,
        )
        sizes = np.bincount(labels, minlength=clusters)
        if sizes.min() < min_cluster_size:
            silhouette = -1.0
        else:
            silhouette = silhouette_score_numpy(normalized, labels)
        score_rows.append(
            {
                "k": clusters,
                "silhouette": round(silhouette, 6),
                "inertia": round(inertia, 6),
                "min_cluster_size": int(sizes.min()),
            }
        )
        if silhouette >= 0 and (best is None or silhouette > best[3]):
            best = (clusters, labels, centers, silhouette)

    if best is None:
        raise ValueError("没有满足最小簇规模的聚类方案")

    clusters, labels, centers, silhouette = best
    cluster_names = _label_clusters(centers)

    grand_mean = normalized.mean(axis=0)
    total_ss = float(((normalized - grand_mean) ** 2).sum())
    within_ss = 0.0
    rows: list[dict[str, object]] = []
    for cluster in range(clusters):
        member_indices = np.flatnonzero(labels == cluster)
        cluster_values = normalized[member_indices]
        centroid = cluster_values.mean(axis=0)
        within_ss += float(((cluster_values - centroid) ** 2).sum())
        actual_profile = values[member_indices].mean(axis=0)
        peak_slot = int(actual_profile.argmax())
        rows.append(
            {
                "cluster_id": cluster,
                "cluster_label": cluster_names[cluster],
                "customers": int(len(member_indices)),
                "mean_daily_gc_kwh": round(
                    float(matrix.iloc[member_indices].mean(axis=1).mean() * 24),
                    6,
                ),
                "peak_slot_start": (
                    pd.Timestamp("2000-01-01")
                    + pd.Timedelta(minutes=30 * peak_slot)
                ).strftime("%H:%M"),
                "normalized_std": round(float(centroid.std()), 6),
                "evening_ratio": round(
                    float(centroid[34:42].mean() / centroid.mean()),
                    6,
                ),
                "morning_ratio": round(
                    float(centroid[10:18].mean() / centroid.mean()),
                    6,
                ),
            }
        )

    explained = 1 - within_ss / total_ss if total_ss > 0 else 0.0
    assignments = pd.DataFrame(
        {
            "customer_id": matrix.index,
            "cluster_id": labels,
        }
    )
    assignments["cluster_label"] = assignments["cluster_id"].map(cluster_names)
    summary = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    metadata = {
        "customers": len(matrix),
        "selected_k": clusters,
        "silhouette": round(float(silhouette), 6),
        "explained_variance_ratio": round(float(explained), 6),
        "k_scores": score_rows,
        "recommend_cluster_feature": bool(
            silhouette >= 0.2 and explained >= 0.2
        ),
    }
    return assignments, summary, metadata


def plot_cluster_profiles(
    matrix: pd.DataFrame,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    clusters = sorted(assignments["cluster_id"].unique())
    columns = 2
    rows = int(np.ceil(len(clusters) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 5.6 if rows == 1 else 4.5 * rows),
        squeeze=False,
    )
    figure.patch.set_facecolor("#F7F8F5")
    hours = np.arange(48) / 2 + 0.25

    for axis, cluster in zip(axes.flat, clusters):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#DDE2DE", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        members = assignments.loc[
            assignments["cluster_id"].eq(cluster), "customer_id"
        ]
        values = matrix.loc[members]
        color = SEASON_COLORS[int(cluster) % len(SEASON_COLORS)]
        for _, member in values.iterrows():
            axis.plot(hours, member.to_numpy(), color="#AAB5B0", alpha=0.15, linewidth=0.8)
        axis.plot(
            hours,
            values.mean(axis=0).to_numpy(),
            color=color,
            linewidth=3.0,
        )
        info = summary.set_index("cluster_id").loc[cluster]
        axis.set_title(
            f"{info['cluster_label']} · {info['customers']} 户 · "
            f"日均 GC {info['mean_daily_gc_kwh']:.1f} kWh",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_ylabel("GC kW")
        format_hour_axis(axis)

    for axis in axes.flat[len(clusters):]:
        axis.axis("off")

    figure.suptitle(
        "300 户工作日常规用电（GC）曲线聚类",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.028,
        "灰线为簇内单户曲线，彩色线为簇平均；聚类基于户内均值归一化后的曲线形态。",
        fontsize=11,
        color="#47545D",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.82 if rows == 1 else 0.9,
        bottom=0.17 if rows == 1 else 0.08,
        hspace=0.35,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    reader = AusgridReader()
    spec = reader.file_specs["Solar home 2010-2011.csv"]
    output_dir = Path(__file__).with_name("reports")
    profile_matrix_path = output_dir / "customer_gc_weekday_profiles.csv"
    if profile_matrix_path.exists():
        matrix = pd.read_csv(profile_matrix_path, index_col=0, encoding="utf-8-sig")
        matrix.columns = matrix.columns.astype(int)
        matrix = matrix.sort_index(axis=1)
    else:
        profiles = collect_customer_profiles(reader, spec)
        matrix = profiles.pivot(
            index="customer_id",
            columns="slot_index",
            values="avg_gc_kw",
        ).sort_index()

    assignments, summary, metadata = cluster_profile_matrix(matrix)
    assignment_path = output_dir / "customer_gc_weekday_clusters.csv"
    assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")

    matrix.to_csv(profile_matrix_path, encoding="utf-8-sig")

    summary_path = output_dir / "customer_gc_weekday_cluster_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "clusters": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure_path = output_dir / "figures" / "customer_gc_weekday_clusters.png"
    plot_cluster_profiles(
        matrix,
        assignments,
        summary,
        output_path=figure_path,
    )

    print(
        f"[OK] 选择 k={metadata['selected_k']}，"
        f"silhouette={metadata['silhouette']:.3f}，"
        f"解释度={metadata['explained_variance_ratio']:.3f}"
    )
    print(summary.to_string(index=False))
    print(f"[OK] 聚类图：{figure_path}")
    print(f"[OK] 客户标签：{assignment_path}")
    print(f"[OK] 聚类汇总：{summary_path}")


if __name__ == "__main__":
    main()
