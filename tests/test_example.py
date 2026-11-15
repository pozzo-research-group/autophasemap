"""example.ipynb as a test, without the plots.

The synthetic phase map has three phases (1, 2 or 3 Gaussian peaks with random
domain warping) on a 12x12 composition grid. The tests check that elastic
k-means recovers the phases and that it beats the Euclidean k-means baseline.
The notebook's BIC sweep over the number of clusters is not included (too slow).

Thresholds have a wide margin over the values observed for seed 0
(elastic ARI 0.92, Euclidean ARI 0.12).
"""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from autophasemap import compute_euclidean_kmeans, multi_kmeans_run

N_CLUSTERS = 3


@pytest.fixture(scope="module")
def elastic_result(data):
    """Notebook cell: "Compute the phasemap using the algorithm presented in this manuscript"."""
    return multi_kmeans_run(
        2,
        data,
        N_CLUSTERS,
        max_iter=50,
        verbose=0,
        smoothen=True,
        threshold=1e-2,
        grid_dim=8,
        random_state=0,
    )


@pytest.fixture(scope="module")
def euclidean_result(data):
    """Notebook cell: "Compare it with Euclidean kmeans clustering as a baseline"."""
    return compute_euclidean_kmeans(data, N_CLUSTERS, max_iter=300, verbose=0, smoothen=True)


def test_synthetic_dataset(data, design):
    points, labels = design
    assert data.N == 144 == points.shape[0]
    assert data.C.shape == (144, 2)
    assert set(np.unique(labels)) == {1, 2, 3}
    assert len(data.F) == data.N
    for f in data.F:
        assert f.shape == (data.n_domain,)
        assert np.all(np.isfinite(f))


def test_elastic_kmeans_outputs(data, elastic_result):
    out, bic = elastic_result
    N, K, D = data.N, N_CLUSTERS, data.n_domain

    assert len(out.templates) == K
    assert all(np.asarray(t).shape == (D,) for t in out.templates)
    assert out.gam_ik.shape == (N, K, D)
    assert out.qik_gam.shape == (N, K, D)
    assert out.fik_gam.shape == (N, K, D)
    assert out.d_amplitude.shape == (N, K)
    assert out.dist.shape == (N, K)
    assert np.asarray(out.delta_n).shape == (N,)

    # constrained assignment guarantees at least two samples per cluster
    counts = np.bincount(out.delta_n, minlength=K)
    assert counts.size == K and counts.min() >= 2

    assert np.isfinite(out.error) and out.error >= 0
    assert 1 <= len(bic) <= 2
    assert np.all(np.isfinite(bic))
    assert np.all(np.isfinite(np.asarray(out.templates)))


def test_elastic_kmeans_recovers_phases(design, elastic_result):
    _, labels = design
    out, _ = elastic_result
    assert adjusted_rand_score(labels, out.delta_n) > 0.8


def test_elastic_beats_euclidean_baseline(data, design, elastic_result, euclidean_result):
    _, labels = design
    out, _ = elastic_result

    assert np.asarray(euclidean_result.delta_n).shape == (data.N,)
    assert euclidean_result.d_amplitude.shape == (data.N, N_CLUSTERS)

    ari_elastic = adjusted_rand_score(labels, out.delta_n)
    ari_euclidean = adjusted_rand_score(labels, euclidean_result.delta_n)
    assert ari_elastic > ari_euclidean + 0.3

