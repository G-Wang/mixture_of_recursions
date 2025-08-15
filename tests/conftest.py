import torch


def pytest_sessionstart(session):
    """Ensure tests run on CPU for faster iteration."""
    torch.set_default_device("cpu")
