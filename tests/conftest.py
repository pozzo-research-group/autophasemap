import logging
import os

import pytest
import ray

from synthetic import ParabolicPhaseDataSet, make_parabolic_design

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Start a local ray instance shared by all tests.

    The dataset class lives in tests/synthetic.py, which ray worker processes
    cannot import unless the tests directory is on their PYTHONPATH.
    """
    # pygsp logs a DEBUG line every time the diffusion graph is built
    logging.getLogger("pygsp").setLevel(logging.WARNING)
    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": TESTS_DIR}},
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="session")
def design():
    """Design points and ground-truth labels of the parabolic phase map."""
    return make_parabolic_design(n_grid=12)


@pytest.fixture(scope="session")
def data(design):
    points, labels = design
    dataset = ParabolicPhaseDataSet(points, labels, n_domain=100, seed=0)
    dataset.generate()
    return dataset
