from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile.newton_pyroki"


def test_newton_pyroki_dockerfile_pins_cudnn_for_jax_cuda():
    dockerfile = DOCKERFILE.read_text()

    assert "ARG CUDNN_VERSION=9.8.0.87" in dockerfile
    assert "nvidia-cudnn-cu12==${CUDNN_VERSION}" in dockerfile
    assert "site-packages/nvidia/cudnn/lib" in dockerfile
    assert "LD_LIBRARY_PATH" in dockerfile
    assert "ENV JAX_PLATFORMS=cuda,cpu" in dockerfile
