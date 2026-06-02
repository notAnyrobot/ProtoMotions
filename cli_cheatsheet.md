```bash
android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ ./scripts/retarget_single_motion_to_robot.sh .venv_isaaclab/bin/python /home/android/miniforge3/envs/pyroki-cuda/bin/python datasets/amass/CMU/13/13_30_poses.motion output/single-motion-test/ astro^C
```

```bash
(pyroki-cuda) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ python pyroki/batch_retarget_from_keypoints.py --keypoints-folder-path datasets/amass/motion_lib/test/keypoints-for-retarget/ --output-dir datasets/dobot/astro/pyroki-retargeted/test/ --source-type smpl --subsample-factor 1 --no-visualize --skip-existing
```

```bash
(pyroki-cuda) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ python pyroki/batch_retarget_from_keypoints.py \
  --robot-type astro \
  --keypoints-folder-path datasets/amass/motion_lib/keypoints-for-retarget/test/ \
  --output-dir datasets/dobot/astro/pyroki-retargeted/test/ \
  --source-type smpl \
  --subsample-factor 1 \
  --no-visualize
```

```bash
(.venv_isaaclab) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ uv run python examples/motion_libs_visualizer.py --motion_files datasets/dobot/astro/proto-astro-amass-test.pt --robot astro --simulator isaaclab
```

```bash
(.venv_isaaclab) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ uv run python examples/motion_libs_visualizer.py --motion_files output/single-motion-test/retargeted_astro_proto/13_13_30_poses_keypoints_retargeted.motion --robot astro --simulator isaaclab
```

```bash
(.venv_isaaclab) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py --retargeted-motion-dir datasets/dobot/astro/pyroki-retargeted/train/ --output-dir datasets/dobot/astro/proto/train/ --robot-type astro --contact-labels-dir datasets/amass/motion_lib/train/contacts/ --apply-motion-filter --force-remake
```

```bash
(.venv_isaaclab) android@ThinkStation-P3-Tower:~/Code/NVlabs/ProtoMotions$ python protomotions/components/motion_lib.py --motion-path datasets/dobot/astro/proto/train/ --output-file datasets/dobot/astro/proto/amass-train.pt
```


```bash
.venv_isaaclab/bin/python protomotions/train_agent.py \
  --robot-name astro \
  --simulator isaaclab \
  --num-envs 4096 \
  --batch-size 16384 \
  --motion-file datasets/dobot/astro/proto/amass-test.pt \
  --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --experiment-name astro-motion-tracker-amass-test
```