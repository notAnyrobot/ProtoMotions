from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "scripts" / "docker" / "README.md"


def test_newton_docker_readme_documents_shell_launchers_and_manual_workflow():
    text = README.read_text()

    assert "./scripts/docker/launch_newton_docker_ws.sh" in text
    assert "./scripts/docker/launch_newton_docker_hpc.sh" in text
    assert "/media/android/data/motion_datasets" in text
    assert "/data/share/motion_datasets" in text
    assert "amass_smpl+hg" in text
    assert "PROTO_DATASET_READONLY=0" in text
    assert "nvidia-smi" in text
    assert "torch.cuda.is_available()" in text
    assert "jax.default_backend()" in text
    assert 'jax.devices("cpu")' in text
    assert "pyroki/batch_retarget_from_keypoints.py" in text
    assert "./scripts/docker/retarget_amass_hpc.sh" in text
    assert "retarget_amass_isaaclab_hpc.sh is a compatibility wrapper" in text
    assert "./scripts/docker/download_retargeted_motion_from_hpc.sh --split sfu" in text
    assert "sfu astro 1" in text
    assert "--robot-type astro" in text
    assert "--keypoints-folder-path" in text
    assert "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py" in text
    assert "protomotions/components/motion_lib.py" in text
    assert "protomotions.train_agent" in text
    assert "--simulator newton" in text
    assert '--motion-file "$DATA/astro/$SPLIT/proto-astro.pt"' in text
    assert '--motion-file "$DATA/astro/proto/amass-$SPLIT.pt"' not in text


def test_newton_docker_readme_documents_rootless_hpc_artifact_cleanup():
    text = README.read_text()

    assert "Rootless HPC artifact cleanup" in text
    assert "stat -c" in text
    assert "namei -l results/smpl-motion-tracker-amass" in text
    assert "container root normally maps to the launching host user" in text
    assert "results/smpl-motion-tracker-amass" in text
    assert "--entrypoint /bin/rm" in text
    assert "Delete and recreate disposable results" in text
    assert "chown -R 0:0 /repo/results" in text
    assert "mkdir -m 775 /repo/results" in text
