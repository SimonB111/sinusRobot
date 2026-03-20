#!/usr/bin/env python3
# Purpose: robot-tracker visualization

# Usage:
# VisualizeRobotTracker.py <marker2gripper_matrix>
# --endoscope2marker_matrix <path_to_endoscope2marker_matrix_txt>

# Arguments:
# marker2gripper_matrix: required, file path to the marker2gripper matrix, formatted as flattened 4x4 homogeneous transformation, space delimited
# --custom_topics <gripper_topic> <gripper_marker_topic> <anatomy_marker_topic>
# --endoscope2marker_matrix <path_to_endoscope2marker_matrix_txt>, optional, formatted as flattened 4x4 homogeneous transformation, space delimited
# --CT_pose <path_to_CT_pose_matrix_txt>, optional, formatted as flattened 4x4 homogeneous transformation, space delimited
# --CT_mesh <path_to_CT_mesh>, optional, path to a valid mesh file
# --mesh_opacity <float 0.0 to 1.0>, optional, specify the level of transparency (1.0 = solid, 0.0 = invisible)

# Output:
# live 3D visualization of gripper (red), marker/tool tip (green),
# tracker (blue), and anatomy (brown) poses, in addition to CT mesh

import rospy
import pyvista as pv

pv.global_theme.multi_samples = 0  # no anti aliasing to drop rendering load
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QTimer
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import PoseStamped
import argparse


class Robot:
    """Class representing a robot"""

    def __init__(
        self,
        targetTopics,
        inputMarker2Gripper: np.array,
        inputEndo2Marker: np.array = np.eye(4),
        inputCTPose: np.array = np.eye(4),
        inputMeshPath: str = "../example/Segmentation_Bone.stl",
        inputMeshOpacity: float = 0.5,
    ) -> None:
        """
        Creates a Robot object
        Parameters:
            targetTopics:
                the 3 topics we need to run visualization:
                gripper, gripper's marker, anatomy marker
            inputMarker2Gripper: required, the homogeneous transformation from
                the marker to the gripper
            inputEndo2Marker: optionally specify a 4x4 calibration matrix from
                tool tip to the marker. Defaults to the identity matrix.
            inputCTPose: optionally specify a 4x4 matrix defining the pose
                of the CT mesh relative to anatomy marker (ct2anatomy)
            inputMeshPath: optionally provide a path to a CT scan mesh.
                Defaults to the example mesh.
            inputMeshOpacity: optionally provide an opacity value from
                0.0 (transparent) to 1.0 (solid). Defaults to 0.5.
        """

        self.gripperTopic = targetTopics[0]
        self.endoMarkerTopic = targetTopics[1]
        self.anatMarkerTopic = targetTopics[2]

        self.gripperPose = None
        self.endoMarkerPose = None
        self.anatPose = None

        # known transformation from the marker to the gripper
        self.marker2gripper = inputMarker2Gripper
        # optionally given transformation
        self.endoscope2marker = inputEndo2Marker
        # optionally given CT pose
        self.CTPose = inputCTPose

        self.lps2ras = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.opacity = inputMeshOpacity

        self.meshPath = inputMeshPath

        self.plotter = BackgroundPlotter()
        self.gripperActor = None
        self.endoActor = None

    def gripperCallback(self, poseIn: PoseStamped) -> None:
        """
        Invoked when receiving data from gripper ros topic,
        updates corresponding pose.
        Returns: None
        """
        self.gripperPose = poseIn

    def endoCallback(self, poseIn: PoseStamped) -> None:
        """
        Invoked when receiving data from endoscope marker ros topic,
        updates corresponding pose.
        Returns: None
        """
        self.endoMarkerPose = poseIn

    def anatCallback(self, poseIn: PoseStamped) -> None:
        """
        Invoked when receiving data from anatomy marker ros topic,
        updates corresponding pose.
        Returns: None
        """
        self.anatPose = poseIn

    def update(self) -> None:
        """
        Calls draw function if we have a valid pose
        """
        if self.gripperPose is not None:
            self.draw()

    def runListeners(self) -> None:
        """
        Creates nodes that listen to needed topics,
        loops continuously
        Returns: None
        """
        # initialize node and subscribe to appropriate topics
        rospy.init_node("listeners", anonymous=True)
        rospy.Subscriber(
            "/REMS/Research/measured_cp", PoseStamped, self.gripperCallback
        )
        rospy.Subscriber(
            "/atracsys/Endoscope/measured_cp", PoseStamped, self.endoCallback
        )
        rospy.Subscriber(
            "/atracsys/Anatomy/measured_cp", PoseStamped, self.anatCallback
        )

        timer = QTimer()
        # schedule task without blocking UI
        timer.timeout.connect(self.update)
        timer.start(65)  # ~15Hz (too fast refresh may freeze sooner)
        self.plotter.app.exec_()

    def poseToHomogeneous(self, poseIn: PoseStamped) -> np.array:
        """
        Turns a given pose into a homogeneous transformation matrix
        Parameters:
            poseIn: PoseStamped, pose we want to turn into a transformation matrix
        Returns:
            output: np.array, the corresponding homogeneous transformation
        """
        pos = poseIn.pose.position
        ori = poseIn.pose.orientation
        # convert quaternion to rot matrix
        rot = Rot.from_quat((ori.x, ori.y, ori.z, ori.w))
        rot_matrix = rot.as_matrix()

        transform = np.eye(4)
        transform[0:3, 0:3] = rot_matrix
        transform[0:3, 3] = np.array([pos.x, pos.y, pos.z])

        return transform

    def transformAxes(self, poseIn: PoseStamped) -> None:
        """
        Creates a rotated and translated version of the standard
        axes to visualize the given pose
        Parameters:
            poseIn: PoseStamped, the pose we want to translate to
        Returns:
            output: pyvista_ndarray, the transformed points
        """
        pos = poseIn.pose.position
        ori = poseIn.pose.orientation

        # convert quaternion to rot matrix
        rot = Rot.from_quat((ori.x, ori.y, ori.z, ori.w))
        rot_matrix = rot.as_matrix()
        # apply rot matrix to points in mesh
        points = self.arrowMeshSave.points.copy()
        rotatedPoints = points.dot(rot_matrix.T)
        return rotatedPoints + np.array([pos.x, pos.y, pos.z])

    def inverse(self, T: np.array) -> np.array:
        """
        Takes the inverse of a 4x4 homogeneous transformation matrix.
        Parameters:
            T: np.array, the homogeneous transformation to be inverted
        Returns:
            T_inv: np.array, the inverted transformation
        """
        R = T[0:3, 0:3]  # rotation part
        t = T[0:3, 3]  # translation part
        R_inv = R.T  # inverse rotation is transpose
        t_inv = -np.dot(R_inv, t)  # inverse translation
        T_inv = np.eye(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = t_inv
        return T_inv

    def applyHomogeneousTransform(
        self, points: pv.pyvista_ndarray, transform
    ) -> pv.pyvista_ndarray:
        """
        Applies a given 4x4 homogeneous transformation matrix to a set of points.
        Parameters:
            points: pv.pyvista_ndarray, points to be transformed
            transform: the 4x4 homogeneous transformation matrix
        Returns:
            output: pv.pyvista_ndarray, the transformed points
        """
        # convert to homogenous points (need correct shape so we can do matrix mult)
        N = points.shape[0]
        homogeneousPoints = np.hstack((points, np.ones((N, 1))))  # now shape (N, 4)

        # (transpose for matrix multiplication), output shape (N,4)
        transformed_homogeneous = (transform @ homogeneousPoints.T).T

        # convert back to 3D by removing the homogeneous coordinate
        transformed_points = (
            transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]
        )
        return transformed_points

    def setupCTMesh(self) -> None:
        """
        Helper function to read in the CT scan mesh from file path,
        add to plotter, set transparency, automatically decimate,
        and scale from mm to m.
        """

        raw_mesh = pv.read(self.meshPath)

        # allow up to maxVert vertices for performance
        # numVert * decFactor = maxVert
        # decFactor = 1 - (maxVert/numVert)
        maxVert = 100000.0
        numVert = raw_mesh.n_points
        decFactor = 1.0 - (maxVert / numVert)

        if decFactor < 1:  # only decimate if needed (more than maxVert verts)
            self.CTMesh = raw_mesh.decimate(decFactor)

        self.CTMeshActor = self.plotter.add_mesh(
            self.CTMesh, color="pink", opacity=float(self.opacity)
        )
        self.CTMesh.scale(0.001, inplace=True)  # scale from mm to m

    def setupCamera(self) -> None:
        """
        Helper function to set up the camera intrinsics
        """
        self.cam = self.plotter.camera
        self.cam.view_angle = 70  # in degrees
        self.cam.clipping_range = (0.001, 1.0) # in meters

    def updateCamera(self, endoscope2base: np.array) -> None:
        """
        Helper function to update camera orientation.
        Parameters:
                endoscope2base: np.array, the homogeneous transformation 
                supplying camera orientation relative to robot base frame
        """
 
        origin = endoscope2base[:3, 3]
        forward_vector = endoscope2base[:3, 2]
        up_vector = endoscope2base[:3, 1]

        # setting camera manually
        self.cam.position = origin
        self.cam.focal_point = origin + forward_vector 
        self.cam.up = up_vector

        # (optional) add an arrow to visualize camera forward direction
        arrow = pv.Arrow(
            start=origin, 
            direction=forward_vector, 
            scale=0.5
        )
        self.plotter.add_mesh(arrow, color="red", name="camera_pointer", 
                            reset_camera=False, opacity=0.2)
        
    def draw(self) -> None:
        """
        Creates a basic 3D visualization of the position and orientation of
        the topics that are subscribed to
        Returns: None
        """
        if self.gripperActor is None:
            # create mesh once
            cubeX = pv.Cube(
                center=(0.01, 0, 0), x_length=0.02, y_length=0.005, z_length=0.005
            )
            cubeY = pv.Cube(
                center=(0, 0.01, 0), x_length=0.005, y_length=0.02, z_length=0.005
            )
            cubeZ = pv.Cube(
                center=(0, 0, 0.01), x_length=0.005, y_length=0.005, z_length=0.3
            )
            self.effectorMesh = pv.merge([cubeX, cubeY, cubeZ])

            # copy meshes and assign to actor for each pose we visualize
            self.arrowMeshSave = self.effectorMesh.copy()
            self.gripperActor = self.plotter.add_mesh(self.effectorMesh, color="red")

            # static axis at (0,0,0) representing the robot base
            self.baseMesh = self.effectorMesh.copy()
            self.plotter.add_mesh(self.baseMesh, color="black")

            self.endoMesh = self.effectorMesh.copy()
            self.endoActor = self.plotter.add_mesh(self.endoMesh, color="green", opacity=.35)

            self.trackerMesh = self.effectorMesh.copy()
            self.trackerActor = self.plotter.add_mesh(self.trackerMesh, color="blue")

            self.anatMesh = self.effectorMesh.copy()
            self.anatActor = self.plotter.add_mesh(self.anatMesh, color="brown")

            self.setupCTMesh() # setup CT mesh one time

            self.setupCamera() # setup camera one time

            self.plotter.show_axes()  # only need to call once
        else:
            # Gripper: apply bTg
            self.effectorMesh.points = self.transformAxes(self.gripperPose)
            # self.gripperActor.user_matrix = self.poseToHomogeneous(self.gripperPose)

            # Endoscope Tip: apply bTe = bTg gTm mTe
            self.gripper2base = self.poseToHomogeneous(self.gripperPose)
            self.endoscope2base = (
                self.gripper2base @ self.marker2gripper @ self.endoscope2marker
            )
            self.endoActor.user_matrix = self.endoscope2base

            # Endoscope Camera: apply bTe (camera is at endoscope tip)
            self.updateCamera(self.endoscope2base)

            # NDI Origin: apply bTT = bTg gTm mTT where (TTm)^-1= mTT
            self.marker2tracker = self.poseToHomogeneous(self.endoMarkerPose)
            self.tracker2base = (
                self.gripper2base
                @ self.marker2gripper
                @ self.inverse(self.marker2tracker)
            )
            self.trackerActor.user_matrix = self.tracker2base

            # NDI Anatomy: apply bTam = bTT TTam
            self.anatMarker2base = self.tracker2base @ self.poseToHomogeneous(
                self.anatPose
            )
            self.anatActor.user_matrix = self.anatMarker2base

            # CT mesh: apply bTct = bTam amTct (calculated above)
            self.ct2base = (
                self.anatMarker2base
                @ self.lps2ras
                @ self.CTPose
                @ self.inverse(self.lps2ras)
            )
            self.CTMeshActor.user_matrix = self.ct2base

        self.plotter.update()  # update the display


if __name__ == "__main__":
    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "marker2gripper_matrix",
        help="required, path to .txt file"
        "containing space delimited 4x4 marker2gripper transformation matrix",
    )
    parser.add_argument(
        "--custom_topics",
        help="provide these 3 ros topics in the following order: "
        "<gripper_topic> <gripper_marker_topic> <anatomy_marker_topic> "
        "(recommended but optional, defaults to "
        "/REMS/Research/measured_cp, /atracsys/Endoscope/measured_cp, /atracsys/Anatomy/measured_cp)",
        nargs=3,
        type=str,
    )
    parser.add_argument(
        "--endoscope2marker_matrix",
        help="provide path to .txt file containing space "
        "delimited 4x4 endoscope2marker transformation matrix (optional)",
    )
    parser.add_argument(
        "--CT_pose",
        help="provide path to .txt containing pose of CT scan mesh"
        " as a space delimited 4x4 transformation matrix (optional)",
    )
    parser.add_argument(
        "--CT_mesh", help="provide path to .stl of the CT scan mesh (optional)"
    )
    parser.add_argument(
        "--mesh_opacity",
        help="specify opacity value from (invisible) 0.0 - 1.0 (solid) (optional)",
    )

    args = parser.parse_args()

    if args.endoscope2marker_matrix:  # handle optional endoscope2marker input
        # read in array
        rawEndo2Marker = np.loadtxt(args.endoscope2marker_matrix)
        # turn flat list into 4x4
        inputEndo2Marker = rawEndo2Marker.reshape(4, 4)
    else:
        inputEndo2Marker = np.eye(4)  # default to identity matrix

    currentTargetTopics = [
        "/REMS/Research/measured_cp",
        "/atracsys/Endoscope/measured_cp",
        "/atracsys/Anatomy_measured_cp",
    ]
    # if we were given custom topics
    if args.custom_topics:
        currentTargetTopics = args.custom_topics

    if args.CT_pose:  # handle optional CT pose input
        rawCTPose = np.loadtxt(args.CT_pose)
        inputCTPose = rawCTPose.reshape(4, 4)
    else:
        inputCTPose = np.eye(4)  # default to identity matrix

    if args.CT_mesh:  # handle optional CT mesh path input
        inputMeshPath = args.CT_mesh
    else:
        inputMeshPath = "../example/Segmentation_Bone.stl"  # default ct mesh

    if args.mesh_opacity:  # handle optional mesh opacity input
        meshOpacity = args.mesh_opacity
    else:
        meshOpacity = 0.5  # default opacity

    # process the required marker2gripper
    rawMarker2Gripper = np.loadtxt(args.marker2gripper_matrix)
    inputMarker2Gripper = rawMarker2Gripper.reshape(4, 4)

    # run nodes and visualization
    sinusRobot = Robot(
        currentTargetTopics,
        inputMarker2Gripper,
        inputEndo2Marker,
        inputCTPose,
        inputMeshPath,
        meshOpacity,
    )
    sinusRobot.runListeners()
