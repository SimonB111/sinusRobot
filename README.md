# sinusRobot

**CalibrateRobotTracker.py** calibrates the gripper2marker homogeneous transformation matrix using OpenCV, extracting the needed data from a .bag file, or by listening to provided topics posting the desired poses.

**VisualizeRobotTracker.py** produces a 3D visualization of the various frames of a surgical robot in the base frame, including the base, optical tracker origin, and the corresponding markers/tip of the tool (endoscope tip here).

This code was tested using the Noetic distro of ROS, using Python 3.8.10, in WSL2: Ubuntu-20.04. Ensure ROS Noetic is installed and call `roscore` prior to live calibration via the listener node.

## CalibrateRobotTracker

Command-line tool to calibrate a robot and tracker by listening to ROS `PoseStamped` topics or a ROS bag file.

### Basic usage

python3 CalibrateRobotTracker.py <output_path>

- `output_path` (required): Path to the output `.txt` file where calibration results will be saved.

### Using custom topics

By default, the script listens to:

- `/REMS/Research/measured_cp` (hand)
- `/NDI/Endoscope/measured_cp` (eye)

To override these with your own topics:

python3 CalibrateRobotTracker.py <output_path>
--custom_topics <hand_topic> <eye_topic>

**Example:**

python3 CalibrateRobotTracker.py results/calibration.txt
--custom_topics /robot/hand_pose /tracker/eye_pose

Each custom topic must publish `geometry_msgs/PoseStamped`.

### Using a ROS bag file

To run calibration on pre-recorded data from a `.bag` file:

python3 CalibrateRobotTracker.py <output_path>
--from_bag <bag_path>

**Example:**

python3 CalibrateRobotTracker.py results/calibration.txt
--from_bag data/session1.bag

You can combine this with custom topics:

python3 CalibrateRobotTracker.py results/calibration.txt
--custom_topics /robot/hand_pose /tracker/eye_pose
--from_bag data/session1.bag

If `--from_bag` is not provided, the script runs ROS listener nodes and waits for live messages on the specified topics.

Note: extractData() attempts to wait for meaningful movement, effectively skipping the first part of a bag file when the robot may be still. However, in some cases the tolerance may need to be adjusted, especially for non-surgical systems were the magnitude is much greater.

## VisualizeRobotTracker
<img width="490" height="322" alt="visualizeRobotFigure" src="https://github.com/user-attachments/assets/808da3df-017e-4d8c-b3c1-bce19519ca4c" />

Command-line tool to visualize robot poses and transformations using PyVista 3D rendering and ROS `PoseStamped` topics.

### Basic usage

python3 RobotVisualizer.py <marker2gripper_matrix>

- `marker2gripper_matrix` (required): Path to `.txt` file with space-delimited 4x4 transformation matrix (marker → gripper).

### Optional endoscope transformation

Provide endoscope-to-marker transformation (defaults to identity matrix):

python3 RobotVisualizer.py <marker2gripper_matrix>
--endoscope2marker_matrix <endoscope2marker_matrix>

**Example:**

python3 RobotVisualizer.py transforms/marker2gripper.txt
--endoscope2marker_matrix transforms/endo2marker.txt

### Input file format

Matrices must be space-delimited in `.txt` files, flattened 4x4 homogeneous transformation, such as:

0.2491029687780142 0.89378931217035 -0.37294554079118875 0.04453415322749031 -0.031509519094885655 -0.3774011125814647 -0.9255136684180745 -0.07724043695033155 -0.9679642871265389 0.2422995370864854 -0.0658488661655321 0.13850916435667032 0.0 0.0 0.0 1.0

### Features

- Live 3D visualization of gripper (red), endoscope (green), tracker (blue), and anatomy (brown) poses
- The green axes represent the marker attached to the gripper if no endoscope2marker is provided, or if the identity matrix was manually provided (matching the default case)
- Static black axes at robot base (0,0,0)
- Automatically subscribes to ROS topics and updates visualization in real-time

The tool runs ROS listener nodes by default and renders pose transformations using the provided matrices.
