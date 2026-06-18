Retargeting SMPL motions with PyRoki
====================================

This workflow covers retargeting SMPL humanoid motions from AMASS to robot morphologies 
(G1 and H1_2) using PyRoki, a trajectory optimization-based retargeting tool.

How Retargeting Works
---------------------

Unlike many common retargeters that solve inverse kinematics (IK) frame-by-frame, 
PyRoki performs **trajectory-level kinematic optimization**. This means:

1. **Whole-trajectory optimization**: Instead of solving each frame independently, 
   PyRoki optimizes the entire motion trajectory at once. This makes it much easier 
   to maintain temporal consistency and smoothness.

2. **No sudden flips**: With our modified PyRoki implementation, we almost never 
   see failures of sudden motion flips and discontinuities.
   This is critical for large-scale data processing and training.

3. **Multiple cost terms**: The optimization balances several objectives simultaneously:

   - **Local alignment** (``local_alignment``): Matches relative joint/keypoint positions 
     and bone directions between source and target
   - **Global alignment** (``global_alignment``): Matches absolute keypoint positions to 
     robot link positions in world frame
   - **Root smoothness** (``root_smoothness``): Penalizes jittery root motion
   - **Joint smoothness** (``joint_smoothness``): Penalizes jittery joint motion
   - **Joint limits** (``limit_cost``): Keeps joints within valid ranges
   - **Joint velocity limits** (``joint_vel_limit``): Prevents unrealistic joint speeds
   - **Foot contact** (``foot_contact``): When feet are in contact, penalizes foot 
     movement and maintains ankle-toe height consistency
   - **Foot tilt** (``foot_tilt``): Keeps feet flat when in contact

4. **Fixed trajectory length**: All motions are trimmed or padded to 15 seconds 
   (450 frames at 30 FPS) for efficient JAX compilation and batch processing.

Overview
--------

The full retargeting pipeline from AMASS to robot:

.. code-block:: text

   Packaged AMASS MotionLib (.pt, SMPL format)
           │
           ▼ (extract_retargeting_input_keypoints_from_packaged_motionlib.py)
   Keypoints (.npy files)
           │
           ├──────────────────────────────────────┐
           ▼                                      ▼
   Retargeted robot motion               Contact labels from source
   (batch_retarget_from_keypoints.py --robot-type {g1,h1_2})
   (--save-contacts-only)
           │                                      │
           └──────────────────────────────────────┘
                           │
                           ▼ (convert_pyroki_retargeted_robot_motions_to_proto.py)
                   ProtoMotions format (.motion)
                           │
                           ▼ (motion_lib.py)
                   Packaged MotionLib (.pt)

Prerequisites
-------------

* Packaged AMASS MotionLib in SMPL format (see :doc:`../../getting_started/amass_preparation`)
* PyRoki installed in a **separate** Python environment (see below)

Installing PyRoki
~~~~~~~~~~~~~~~~~

PyRoki requires a separate Python environment from ProtoMotions due to different 
JAX/CUDA dependencies. Install it as follows:

.. code-block:: bash

   # Create a new environment for PyRoki
   conda create -n pyroki python=3.10
   conda activate pyroki
   
   # Clone and install PyRoki
   git clone https://github.com/chungmin99/pyroki.git
   cd pyroki
   pip install -e .

For more details, see the `PyRoki GitHub repository <https://github.com/chungmin99/pyroki>`_.

Quick Start: Convenience Script
-------------------------------

For a one-click solution, use the provided bash script. Since ProtoMotions and 
PyRoki require separate Python environments, you must provide paths to both 
Python interpreters:

.. code-block:: bash

   ./scripts/retarget_amass_to_robot.sh <proto_python> <pyroki_python> <amass_pt_file> <robot_type> [skip_freq]

**Arguments:**

* ``proto_python``: Path to Python interpreter with ProtoMotions installed
* ``pyroki_python``: Path to Python interpreter with PyRoki installed
* ``amass_pt_file``: Path to packaged AMASS MotionLib .pt file under ``smpl/<split>/``
* ``robot_type``: Target robot (``g1``, ``h1_2``, or ``astro``)
* ``skip_freq``: (Optional) Skip every N motions (default: 1 = all motions)

**Example:**

.. code-block:: bash

   # Retarget every 50th motion to G1 (for quick testing)
   ./scripts/retarget_amass_to_robot.sh \
       ~/miniconda3/envs/protomotions/bin/python \
       ~/miniconda3/envs/pyroki/bin/python \
       /data/protomotions/smpl/train/amass_smpl_train.pt \
       g1 50

   # Retarget all motions to Astro
   ./scripts/retarget_amass_to_robot.sh \
       ~/miniconda3/envs/protomotions/bin/python \
       ~/miniconda3/envs/pyroki/bin/python \
       /data/protomotions/smpl/train/amass_smpl_train.pt \
       astro 1

The script runs all steps automatically and outputs the final MotionLib ``.pt`` file.

**Output folder structure** (SMPL-derived keypoints and contacts are shared across robots; robot outputs are saved per robot and split):

.. code-block:: text

   /data/protomotions/
   ├── smpl/
   │   └── train/
   │       ├── amass_smpl_train.pt
   │       ├── keypoints-for-retarget/   # Keypoints extracted from SMPL
   │       └── contacts/                 # Foot contact labels from source
   └── astro/
       └── train/
           ├── pyroki-retargeted-astro/  # Retargeted robot motions
           ├── proto-astro/              # Robot proto format motions
           └── proto-astro.pt            # Packaged robot MotionLib

Retargeting a Single Motion File
--------------------------------

To retarget a single ``.motion`` file (instead of a packaged ``.pt`` MotionLib), use 
the dedicated script:

.. code-block:: bash

   ./scripts/retarget_single_motion_to_robot.sh <proto_python> <pyroki_python> <motion_file> <output_dir> <robot_type>

**Arguments:**

* ``proto_python``: Path to Python interpreter with ProtoMotions installed
* ``pyroki_python``: Path to Python interpreter with PyRoki installed
* ``motion_file``: Path to input ``.motion`` file (SMPL format)
* ``output_dir``: Directory for all outputs
* ``robot_type``: Target robot (``g1`` or ``h1_2``)

**Example:**

.. code-block:: bash

   ./scripts/retarget_single_motion_to_robot.sh \
       ~/miniconda3/envs/protomotions/bin/python \
       ~/miniconda3/envs/pyroki/bin/python \
       /path/to/walk.motion \
       /path/to/output \
       g1

The script automatically:

1. Extracts keypoints from the SMPL motion
2. Runs PyRoki retargeting to the target robot
3. Extracts foot contact labels from the source motion
4. Converts to ProtoMotions format with contacts
5. Reports the output ``.motion`` file path

To visualize the result:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/output/proto-g1.pt \
       --robot g1 \
       --simulator isaacgym

Step-by-Step Guide
------------------

Step 1: Extract Keypoints from Packaged MotionLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract simplified keypoints (pelvis, shoulders, elbows, wrists, hips, knees, 
ankles, feet, plus auxiliary points) from the packaged SMPL motions:

.. code-block:: bash

   python data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
       /path/to/amass_train.pt \
       --output-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --skeleton-format smpl \
       --start-idx 0 \
       --skip-freq 15

**Arguments:**

* ``--output-path``: Directory for extracted keypoint ``.npy`` files
* ``--skeleton-format``: Source skeleton format (``smpl`` for AMASS)
* ``--start-idx``: Starting motion index (default: 0)
* ``--skip-freq``: Skip every N motions (use 15-35 for quick subset testing, 1 for all motions)

.. tip::

   Use ``--skip-freq 50`` or higher when first testing the pipeline to process 
   only a small subset of motions. Once verified, set ``--skip-freq 1`` to 
   process all motions.

Step 2: Run PyRoki Retargeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the PyRoki environment (separate from ProtoMotions) and run the canonical
batch retargeting CLI:

.. code-block:: bash

   conda activate pyroki  # Switch to PyRoki environment

   python pyroki/batch_retarget_from_keypoints.py \
       --robot-type g1 \
       --keypoints-folder-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --output-dir /tmp/protomotions-retarget/pyroki-retargeted-g1/ \
       --source-type smpl \
       --subsample-factor 1 \
       --no-visualize \
       --skip-existing

Use ``--robot-type h1_2`` and an H1_2 output directory for H1_2 retargeting.

**Arguments:**

* ``--robot-type``: Target robot (``g1`` or ``h1_2``)
* ``--keypoints-folder-path``: Input directory with keypoint ``.npy`` files
* ``--output-dir``: Output directory for retargeted motions (``.npz`` files)
* ``--source-type``: Source skeleton type (``smpl`` for AMASS, ``rigv1`` for custom rigs)
* ``--subsample-factor``: Temporal subsampling (1 = no subsampling)
* ``--no-visualize``: Skip visualization (required for batch processing)
* ``--skip-existing``: Resume interrupted runs by skipping completed files

Step 3: Extract Contact Labels from Source Motions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Foot contact labels should come from the **source SMPL motions**, not re-computed 
from retargeted robot motions. This is because the retargeting process can be 
imperfect, and source motion contacts are more reliable.

.. code-block:: bash

   python pyroki/batch_retarget_from_keypoints.py \
       --robot-type g1 \
       --keypoints-folder-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --source-type smpl \
       --subsample-factor 1 \
       --save-contacts-only \
       --contacts-dir /tmp/protomotions-retarget/contacts/ \
       --skip-existing

Use the same ``--robot-type`` value used for retargeting. Contact extraction is based
on source keypoints and writes the same ``*_contacts.npz`` schema for each robot.

Step 4: Convert to ProtoMotions Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert retargeted motions to ProtoMotions format, incorporating the source contact labels:

.. code-block:: bash

   python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
       --retargeted-motion-dir /tmp/protomotions-retarget/pyroki-retargeted-g1/ \
       --output-dir /tmp/protomotions-retarget/proto-g1/ \
       --robot-type g1 \
       --contact-labels-dir /tmp/protomotions-retarget/contacts/ \
       --apply-motion-filter \
       --force-remake

**Arguments:**

* ``--retargeted-motion-dir``: Directory with retargeted ``.npz`` files
* ``--output-dir``: Output directory for ``.motion`` files
* ``--robot-type``: Target robot (``g1`` or ``h1_2``)
* ``--contact-labels-dir``: Directory with contact labels from Step 3
* ``--apply-motion-filter``: Apply smoothing filter to reduce jitter
* ``--force-remake``: Overwrite existing files

.. note::

   The conversion script automatically adjusts the robot height (``fix_height``) to 
   ensure feet don't penetrate the ground, using robot-specific foot offsets.

Step 5: Package into MotionLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package the converted motions into a single ``.pt`` file:

.. code-block:: bash

   python protomotions/components/motion_lib.py \
       --motion-path /tmp/protomotions-retarget/proto-g1/ \
       --output-file /tmp/protomotions-retarget/proto-g1.pt

Step 6: Verify with Motion Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before training, verify the retargeted motions look correct using the motion 
visualizer:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /tmp/protomotions-retarget/proto-g1.pt \
       --robot g1 \
       --simulator isaacgym

The visualizer supports comparing multiple MotionLibs side-by-side:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /tmp/protomotions-retarget/proto-g1.pt /path/to/reference.pt \
       --robot g1 \
       --simulator isaacgym

.. image:: ../../_static/motion_libs_viz.png
   :width: 600
   :alt: Motion Libs Visualizer

**Controls:**

* **R**: Switch to next motion
* **1/2**: Increase/decrease playback speed
* **3/4**: Adjust smoothness threshold for highlighting

Adding a New Robot for Retargeting
----------------------------------

To add PyRoki retargeting support for a new robot, add a config under
``pyroki/retargeting/configs/`` and register it in
``pyroki/retargeting/factory.py``. This is enough for the PyRoki retargeting
stage only. Full ProtoMotions pipeline support also needs ProtoMotions robot
config and asset support, conversion support in
``convert_pyroki_retargeted_robot_motions_to_proto.py``, and updates to the
convenience-script robot allowlists. New robot support should not copy the
solver script.

The config owns:

* ``robot_type`` and ``display_name``
* retargeting URDF path and mesh directory
* source keypoint to robot link mapping
* source-type scaling for ``smpl`` and ``rigv1``
* optimization weights
* hand and torso auxiliary offsets
* global-alignment keypoint weights such as hip or elbow downweighting
* short robot-specific hooks when a behavior cannot be represented as fields

The shared solver owns file discovery, keypoint loading, contact writing, PyRoki
optimization, visualization, and ``*_retargeted.npz`` output writing.

Compatibility Wrappers
----------------------

The old robot-specific scripts still exist for temporary compatibility:

.. code-block:: bash

   python pyroki/batch_retarget_to_g1_from_keypoints.py \
       --keypoints-folder-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --source-type smpl \
       --no-visualize
   python pyroki/batch_retarget_to_h1_2_from_keypoints.py \
       --keypoints-folder-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --source-type smpl \
       --no-visualize

They delegate to the canonical CLI and emit a deprecation warning. New automation
should call:

.. code-block:: bash

   python pyroki/batch_retarget_from_keypoints.py \
       --robot-type g1 \
       --keypoints-folder-path /tmp/protomotions-retarget/keypoints-for-retarget/ \
       --source-type smpl \
       --no-visualize

Next Steps
----------

* :doc:`../../getting_started/amass_preparation` - Prepare AMASS data
* :doc:`amass_smpl` - Train SMPL policy on AMASS
* :doc:`custom_robot` - Add your own robot to ProtoMotions
