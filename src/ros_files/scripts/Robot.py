#!/usr/bin/env python

import rospy
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import PoseStamped
import argparse

class Robot:
    '''Class representing a robot'''

    def __init__(self, calibMatrix: np.array = np.eye(4)) -> None:
        '''
        Creates a Robot object
        Parameters:
            calibMatrix: optionally specify a 4x4 calibration matrix.
                         Defaults to the identity matrix.
        '''
        self.gripperPose = None
        self.endoMarkerPose = None
        self.anatPose = None

        self.handEyeIsCalibrated = False
        self.sampleCount = 0
        self.maxSamples = 200
        self.tolerance = 0.06 # usually around .1, in seconds
        # allocate arrays with appropriate shape
        self.tHand = np.zeros((self.maxSamples, 3)) 
        self.tEye = np.zeros((self.maxSamples, 3))
        self.rHand = np.zeros((self.maxSamples, 3, 3))
        self.rEye = np.zeros((self.maxSamples, 3, 3))
        # to be filled by calibrate function
        self.marker2gripper = np.eye(4) 
        # known transformation
        self.endoscope2marker = calibMatrix

        self.plotter = BackgroundPlotter()
        self.gripperActor = None
        self.endoActor = None
        
    def gripperCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from /REMS/Research/measured_cp,
        updates corresponding pose.
        Returns: None
        '''
        self.gripperPose = poseIn

    def endoCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from NDI/Endoscope/measured_cp,
        updates corresponding pose. Runs collectCalibData if not calibrated yet
        Returns: None
        '''
        self.endoMarkerPose = poseIn
        if not self.handEyeIsCalibrated:
            self.collectCalibData()
    
    def anatCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from NDI/Endoscope/measured_cp,
        updates corresponding pose.
        Returns: None
        '''
        self.anatPose = poseIn

    def collectCalibData(self) -> None:
        '''
        Collects calibration data if they are within time difference tolerance

        '''
        # measure the difference in times
        # collect that data pair if time difference within our tolerance
        gripperTime = self.gripperPose.header.stamp.to_sec()
        endoMarkerTime = self.endoMarkerPose.header.stamp.to_sec()
        diff = abs(gripperTime - endoMarkerTime)
        if diff < self.tolerance:
            rospy.loginfo(f"{self.sampleCount} of {self.maxSamples} samples, time diff = {diff}")
            self.collectHandEye(self.gripperPose, self.endoMarkerPose)

    def update(self) -> None:
        '''
        Calls draw function if we have a valid pose
        '''
        if self.gripperPose is not None:
            self.draw()

    def runListeners(self) -> None:
        '''
        Creates nodes that listen to needed topics,
        loops continuously
        Returns: None
        '''
        # initialize node and subscribe to appropriate topics
        rospy.init_node('listeners', anonymous=True)
        rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, self.gripperCallback)
        rospy.Subscriber("/NDI/Endoscope/measured_cp", PoseStamped, self.endoCallback)
        rospy.Subscriber("/NDI/Anatomy/measured_cp", PoseStamped, self.anatCallback)

        timer = QTimer()
        # schedule task without blocking UI
        timer.timeout.connect(self.update)
        timer.start(65) # ~15Hz (too fast refresh freezes sooner)
        self.plotter.app.exec_()

    def collectHandEye(self, gripperPose: PoseStamped, markerPose: PoseStamped):
        '''
        Collects arrays full of translation vectors and
        rotation matrices for gripper2base and target2cam
        Calls calibrateHandEye() when full
        forms self.marker2gripper
        Parameters:
            gripperPose: PoseStamped, representing gripper2base
            markerPose: PoseStamped, representing target2cam (marker2NDI)
        '''
        if self.sampleCount < self.maxSamples:
            hPos = gripperPose.pose.position
            hOri = gripperPose.pose.orientation
            ePos = markerPose.pose.position
            eOri = markerPose.pose.orientation

            # turn quat to rot matrix and assign for rHand and rEye
            hRot = Rot.from_quat((hOri.x, hOri.y, hOri.z, hOri.w))
            self.rHand[self.sampleCount] = hRot.as_matrix()

            eRot = Rot.from_quat((eOri.x, eOri.y, eOri.z, eOri.w))
            invEyeRot = eRot.as_matrix().T # invert eye
            self.rEye[self.sampleCount] = invEyeRot

            # fill current row with position vector
            self.tHand[self.sampleCount, :] = [hPos.x, hPos.y, hPos.z]
            # invert eye
            invEyeTranslation = invEyeRot @ -np.array([ePos.x, ePos.y, ePos.z])
            self.tEye[self.sampleCount, :] = invEyeTranslation
            
            self.sampleCount += 1 # move to next position
        else: # call calibrate when we have all samples
            rMarker2Gripper, tMarker2Gripper = cv2.calibrateHandEye(
                self.rHand, self.tHand, self.rEye, self.tEye)
            
            self.marker2gripper[:3, :3] = rMarker2Gripper # rotation part
            self.marker2gripper[:3, 3] = tMarker2Gripper.flatten() # translation part

            rospy.loginfo(self.marker2gripper)

            self.handEyeIsCalibrated = True

    def poseToHomogeneous(self, poseIn: PoseStamped) -> np.array:
        '''
        Turns a given pose into a homogeneous transformation matrix
        Parameters:
            poseIn: PoseStamped, pose we want to turn into a transformation matrix
        Returns:
            output: np.array, the corresponding homogeneous transformation
        '''
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
        '''
        Creates a rotated and translated version of the standard 
        axes to visualize the given pose
        Parameters:
            poseIn: PoseStamped, the pose we want to translate to
        Returns: 
            output: pyvista_ndarray, the transformed points
        '''
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
        '''
        Takes the inverse of a 4x4 homogeneous transformation matrix.
        Parameters:
            T: np.array, the homogeneous transformation to be inverted
        Returns:
            T_inv: np.array, the inverted transformation
        '''
        R = T[0:3, 0:3]            # rotation part
        t = T[0:3, 3]              # translation part
        R_inv = R.T                # inverse rotation is transpose
        t_inv = -np.dot(R_inv, t)  # inverse translation
        T_inv = np.eye(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = t_inv
        return T_inv

    def applyHomogeneousTransform(self, points: pv.pyvista_ndarray, 
                                 transform) -> pv.pyvista_ndarray:
        '''
        Applies a given 4x4 homogeneous transformation matrix to a set of points.
        Parameters:
            points: pv.pyvista_ndarray, points to be transformed
            transform: the 4x4 homogeneous transformation matrix
        Returns: 
            output: pv.pyvista_ndarray, the transformed points
        '''
        # convert to homogenous points (need correct shape so we can do matrix mult)
        N = points.shape[0]
        homogeneousPoints = np.hstack((points, np.ones((N, 1)))) # now shape (N, 4)

        # (transpose for matrix multiplication), output shape (N,4)
        transformed_homogeneous = (transform @ homogeneousPoints.T).T  

        # convert back to 3D by removing the homogeneous coordinate
        transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]
        return transformed_points

    def draw(self) -> None:
        '''
        Creates a basic 3D visualization of the position and orientation of
        the topics that are subscribed to
        Returns: None
        '''
        if self.gripperActor is None:
            # create mesh once
            cubeX = pv.Cube(center=(.25,0,0), x_length=.5, y_length=0.05, z_length=0.05)
            cubeY = pv.Cube(center=(0,.25,0), x_length=0.05, y_length=.5, z_length=0.05)
            cubeZ = pv.Cube(center=(0,0,.25), x_length=0.05, y_length=0.05, z_length=.5)
            # combine meshes
            self.effectorMesh = pv.merge([cubeX, cubeY, cubeZ])

            # copy meshes and assign to actor for each pose we visualize
            self.arrowMeshSave = self.effectorMesh.copy()
            self.gripperActor = self.plotter.add_mesh(self.effectorMesh, color='red')

            # static axis at (0,0,0) representing the robot base
            self.baseMesh = self.effectorMesh.copy()
            self.plotter.add_mesh(self.baseMesh, color='black') 

            self.endoMesh = self.effectorMesh.copy()
            self.endoActor = self.plotter.add_mesh(self.endoMesh, color='green')

            self.trackerMesh = self.effectorMesh.copy()
            self.trackerActor = self.plotter.add_mesh(self.trackerMesh, color='blue')
            
            self.anatMesh = self.effectorMesh.copy()
            self.anatActor = self.plotter.add_mesh(self.anatMesh, color='brown')

            self.plotter.show_axes() # only need to call once
        else :
            # directly apply pose transformation for gripper
            self.effectorMesh.points = self.transformAxes(self.gripperPose)

            # Endoscope Tip: apply bTe = bTg gTm mTe
            self.gripper2base = self.poseToHomogeneous(self.gripperPose)
            self.endoscope2base = (self.gripper2base @ self.marker2gripper 
                                   @ self.endoscope2marker)
            self.endoMesh.points = self.applyHomogeneousTransform(
                self.arrowMeshSave.points.copy(), self.endoscope2base)
            
            # NDI Origin: apply bTT = bTg gTm mTT where (TTm)^-1= mTT
            self.marker2tracker = self.poseToHomogeneous(self.endoMarkerPose)
            self.tracker2base = (self.gripper2base @ self.marker2gripper 
                                 @ self.inverse(self.marker2tracker))
            self.trackerMesh.points = self.applyHomogeneousTransform(
                self.arrowMeshSave.points.copy(), self.tracker2base)
            
            # NDI Anatomy: apply bTam = bTT TTam
            self.anatMarker2base = (self.tracker2base 
                                    @ self.poseToHomogeneous(self.anatPose))
            self.anatMesh.points = self.applyHomogeneousTransform(
                self.arrowMeshSave.points.copy(), self.anatMarker2base)

        self.plotter.update() # update the display
  

if __name__ == '__main__':
    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_calib", 
                        help="provide path to .txt file containing space delimited 4x4 calib matrix")
    args = parser.parse_args()

    # if we were given an input
    if args.camera_calib:
        # read in array
        flatInputMatrix = np.loadtxt(args.camera_calib)
        # turn flat list into 4x4
        inputMatrix = flatInputMatrix.reshape(4, 4)
        rospy.loginfo(inputMatrix)
    else:
        inputMatrix = np.eye(4) # default to identity matrix

    # run nodes and visualization
    sinusRobot = Robot(inputMatrix)
    sinusRobot.runListeners()