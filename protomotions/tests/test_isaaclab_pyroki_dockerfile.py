from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile.isaaclab_pyroki"


def test_isaaclab_pyroki_dockerfile_installs_full_jax_cuda_extra():
    dockerfile = DOCKERFILE.read_text()

    assert "FROM protomotions-isaaclab:2.3.0" in dockerfile
    assert 'RUN uv venv --python python3 --seed --index-url "${PIP_INDEX_URL}" "$PYROKI_VENV"' in dockerfile
    assert '"jax[cuda12]==0.6.2"' in dockerfile
    assert '"jax-cuda12-plugin==0.6.2"' not in dockerfile
    assert '"jax-cuda12-pjrt==0.6.2"' not in dockerfile
    assert "ARG CUDNN_VERSION=9.8.0.87" in dockerfile
    assert "nvidia-cudnn-cu12==${CUDNN_VERSION}" in dockerfile

    assert "site-packages/nvidia/cudnn/lib" in dockerfile
    assert "ENV LD_LIBRARY_PATH=" in dockerfile

    assert "PYROKI_JAX_CUDA_LIB" not in dockerfile
    assert "lib/jax-cuda" not in dockerfile
    assert "No NVIDIA CUDA wheel library directories found" not in dockerfile


def test_isaaclab_pyroki_dockerfile_preserves_separate_python_environments():
    dockerfile = DOCKERFILE.read_text()

    assert "ENV PYROKI_PYTHON=/workspace/pyroki-venv/bin/python" in dockerfile
    assert "python3 -m venv" not in dockerfile
    assert "PATH=/workspace/pyroki-venv/bin" not in dockerfile
