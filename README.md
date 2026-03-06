# sinusRobot

https://github.com/user-attachments/assets/d3ff68a3-32b9-4695-a259-b45feb5d314a

**CalibrateRobotTracker.py** calibrates the marker2gripper homogeneous transformation matrix using OpenCV, extracting the needed data from a .bag file, or by listening to provided topics posting the desired poses.

**VisualizeRobotTracker.py** produces a 3D visualization of the various frames of a surgical robot in the base frame, including the base, optical tracker origin, and the corresponding markers/tip of the tool (endoscope tip here).

This code was tested using the Noetic distro of ROS, using Python 3.8.10, in WSL2: Ubuntu-20.04. Ensure ROS Noetic is installed and call `roscore` prior to live calibration via the listener node.

## CalibrateRobotTracker

Command-line tool to calibrate a robot and tracker by listening to ROS `PoseStamped` topics or a ROS bag file.

### Basic usage

python3 CalibrateRobotTracker.py <output_path>

- `output_path` (required): Path to the output `.txt` file where calibration results will be saved.

### Options

- `--custom_topics <hand_topic> <eye_topic>`: Override default topics (`/REMS/Research/measured_cp` for hand, `/atracsys/Endoscope/measured_cp` for eye).
- `--from_bag <bag_path>`: Use data from a ROS bag file instead of live topics.
- `--max_samples <number>`: Set maximum samples to collect (default: 400).

**Example:**

python3 CalibrateRobotTracker.py results/calibration.txt --from_bag data/session1.bag --max_samples 500

## VisualizeRobotTracker

Command-line tool to visualize robot poses and transformations using PyVista 3D rendering and ROS `PoseStamped` topics.

<img width="541" height="322" alt="visualizeRobotFigure_surgicalScale_labeled" src="https://github.com/user-attachments/assets/6f5e0569-1946-44f6-ae92-91403ae5743d" />
(2cm surgical scale axes)

### Basic usage

python3 VisualizeRobotTracker.py <marker2gripper_matrix>

- `marker2gripper_matrix` (required): Path to `.txt` file with space-delimited 4x4 transformation matrix (marker → gripper).

### Options

- `--custom_topics <gripper_topic> <gripper_marker_topic> <anatomy_marker_topic>`: Override default topics.
- `--endoscope2marker_matrix <path>`: Path to endoscope-to-marker transformation matrix (optional).
- `--CT_pose <path>`: Path to CT scan pose matrix (optional).
- `--CT_mesh <path>`: Path to CT mesh file (optional, defaults to `../example/Segmentation_Bone.stl`).
- `--mesh_opacity <float>`: Opacity value (0.0-1.0, default: 0.5).

**Example:**

python3 VisualizeRobotTracker.py transforms/marker2gripper.txt --CT_mesh model.stl --mesh_opacity 0.7

### Input file format

Matrices must be space-delimited in `.txt` files, flattened 4x4 homogeneous transformation, such as:

`0.2491029687780142 0.89378931217035 -0.37294554079118875 0.04453415322749031 -0.031509519094885655 -0.3774011125814647 -0.9255136684180745 -0.07724043695033155 -0.9679642871265389 0.2422995370864854 -0.0658488661655321 0.13850916435667032 0.0 0.0 0.0 1.0`

### Features

- Live 3D visualization of gripper (red), endoscope (green), tracker (blue), and anatomy (brown) poses
- The green axes represent the marker attached to the gripper if no endoscope2marker is provided, or if the identity matrix was manually provided (matching the default case)
- Static black axes at robot base (0,0,0)
- Automatically subscribes to ROS topics and updates visualization in real-time

The tool runs ROS listener nodes by default and renders pose transformations using the provided matrices.
