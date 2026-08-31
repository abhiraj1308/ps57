import numpy as np

def denoise(signal, window=5):
    """Moving average filter — smooths out background noise."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

def detect_targets(signal, threshold=0.3, min_gap=3):
    """Finds points above threshold, then groups nearby points into
    a single target instead of reporting each point separately."""
    above = [(i, float(v)) for i, v in enumerate(signal) if v > threshold]
    if not above:
        return []

    targets = []
    cluster = [above[0]]
    for point in above[1:]:
        if point[0] - cluster[-1][0] <= min_gap:
            cluster.append(point)
        else:
            targets.append(_summarize(cluster))
            cluster = [point]
    targets.append(_summarize(cluster))
    return targets

def _summarize(cluster):
    peak = max(cluster, key=lambda p: p[1])
    indices = [p[0] for p in cluster]
    return {
        "range_index": peak[0],
        "intensity": peak[1],
        "span": (min(indices), max(indices)),
    }