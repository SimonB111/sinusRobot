#!/usr/bin/env python3
    # Purpose: robot-tracker calibration

    # Usage:
    # CalibrateRobotTracker.py <output_path> 
        # --custom_topics <hand_topic> <eye_topic> 
        # --from_bag <bag_path>  

    # Arguments:
    # output_path: REQUIRED, file path for the marker2gripper matrix
    # --custom_topics <hand_topic> <eye_topic>: hand/eye rostopic paths
    # --from_bag: file path to the .bag to extract calibration data from (will not run nodes for live calibration)

    # Output:
    # <output_path>: 4x4 homogeneous marker2gripper matrix, flattened, delimited by spaces

import rospy
import rosbag
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import PoseStamped
import argparse

class CalibrateRobotTracker:
    '''
    Class to collect data then calibrate handeye matrix
    '''

    def __init__(self, outputPath: str, targetTopics) -> None:
        '''
        Creates a CalibrateRobotTracker object
        Parameters:
            outputPath: string, the path to the output txt file
            targetTopics: list of two strings, format: ["/hand/path", "/eye/path"]
        '''
        self.outputPath = outputPath

        self.handEyeIsCalibrated = False
        self.sampleCount = 0
        self.maxSamples = 150

        # track when meaningful movement starts to avoid near-static data
        # low pose-diversity will cause innacurate calibration
        self.tolerance = 0.06 # in seconds
        self.distThreshold = 0.0059 # move at least this much (meters) to trigger collection
        self.startedMoving = False  
        self.lastPos = None
        
        self.startedMoving = False  
        self.forceCalibrate = False
        self.targetTopics = targetTopics
        # allocate arrays with appropriate shape
        self.tHand = np.zeros((self.maxSamples, 3)) 
        self.tEye = np.zeros((self.maxSamples, 3))
        self.rHand = np.zeros((self.maxSamples, 3, 3))
        self.rEye = np.zeros((self.maxSamples, 3, 3))
        # to be filled by calibrate function
        self.marker2gripper = np.eye(4) 
        
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
    
    def collectCalibData(self) -> None:
        '''
        Collects calibration data if pair is within time difference tolerance

        '''
        # measure the difference in times
        # collect that data pair if time difference within our tolerance
        gripperTime = self.gripperPose.header.stamp.to_sec()
        endoMarkerTime = self.endoMarkerPose.header.stamp.to_sec()
        diff = abs(gripperTime - endoMarkerTime)
        if diff < self.tolerance:
            print(f"{self.sampleCount} of {self.maxSamples} samples, time diff = {diff}")
            self.collectHandEye(self.gripperPose, self.endoMarkerPose)

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
        # stay running
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.handEyeIsCalibrated:
                rospy.signal_shutdown("Calibration Complete")
                break
            rate.sleep()

    def detectMotion(self, topic, pos) -> None:
        '''
        Helper function to check for meaningful movement since last pose
        Parameters:
            topic: topic of the current message
            pos: position part of PoseStamped of current position
        '''
        # detect meaningful movement from hand before collecting data
        if topic == self.targetTopics[0] and self.lastPos == None:
            self.lastPos = pos # get initial pos
        elif topic == self.targetTopics[0]:
            # find distance traveled in space by hand
            dist = np.sqrt(  (pos.x - self.lastPos.x)**2 
                            + (pos.y - self.lastPos.y)**2 
                            + (pos.z - self.lastPos.z)**2 )
            if dist > self.distThreshold:
                self.startedMoving = True

    def extractData(self, bagPath):
        '''
        Calls collectHandEye with pairs of hand and eye data that are within the
        time tolerance limit until calibration is done, using data from the
        provided .bag file. 
        The function is designed to calculate marker2gripper, where the marker is 
        connected to the moving gripper, and the camera/tracker is stationary.
        '''
      
        # lists to store our extracted PoseStamped for hand and eye
        handPoses = []
        eyePoses = []
        usedPoses = 0
        with rosbag.Bag(bagPath, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=self.targetTopics):
                pos = msg.pose.position

                # check for meaningful movement from hand before collecting data
                if not self.startedMoving: 
                    self.detectMotion(topic, pos)

                # pack data into two lists
                if self.startedMoving:
                    if topic == self.targetTopics[0]:
                        handPoses.append(msg)
                    elif topic == self.targetTopics[1]:
                        eyePoses.append(msg)
                    usedPoses += 1

                # place limit to prevent excessive time and space usage
                # for large .bag files
                if usedPoses > self.maxSamples*200:
                    break
        
        hI = 0 # hand index
        eI = 0 # eye index
        while not self.handEyeIsCalibrated and hI < len(handPoses) and eI < len(eyePoses):
            handTime = handPoses[hI].header.stamp.to_sec()
            eyeTime = eyePoses[eI].header.stamp.to_sec()
            diff = abs(handTime - eyeTime)

            # if we hit a final iteration without calibration,
            # then force calibration with existing data
            if hI == len(handPoses)-1 or eI == len(eyePoses)-1:
                self.forceCalibrate = True
                print(f"forcing calibration, # of hand poses = {len(handPoses)}, hI = {hI}, eI = {eI}")
                self.collectHandEye(handPoses[hI], eyePoses[eI])
                break

            # found good match
            elif diff < self.tolerance:
                self.collectHandEye(handPoses[hI], eyePoses[eI])
                # advance both to look for a new pair OPTIONAL
                hI += 1
                eI += 1
                continue

            if handTime < eyeTime:
                hI += 1 # incrementing hand first since its faster refresh
            elif eyeTime < handTime:
                eI += 1

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
        if self.sampleCount < self.maxSamples and not self.forceCalibrate:
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
                self.rHand, self.tHand, self.rEye, self.tEye, cv2.CALIB_HAND_EYE_PARK)
            
            self.marker2gripper[:3, :3] = rMarker2Gripper # rotation part
            self.marker2gripper[:3, 3] = tMarker2Gripper.flatten() # translation part

            # write out as space separated numbers
            with open(self.outputPath, 'w') as f:
                f.write(' '.join(map(str, self.marker2gripper.flatten())))

            print(self.marker2gripper)
            self.handEyeIsCalibrated = True


if __name__ == '__main__':
    # usage:
    # CalibrateRobotTracker.py <output_path> --custom_topics <hand_topic> <eye_topic> --from_bag <bag_path>  
    # output path is required, the rest is optional

    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", default="output.txt", help="Required, path for output .txt")
    parser.add_argument("--custom_topics", nargs=2, type=str,
                        help="Provide paths to hand, then eye rostopic. Each" \
                        "should post PoseStamped")
    parser.add_argument("--from_bag", 
                        help="Provide path to a .bag file containing appropriate topics. " \
                        "If no path is provided, the program will run a listener node for the topics")
    args = parser.parse_args()

    currentTargetTopics = ["/REMS/Research/measured_cp", 
                           "/NDI/Endoscope/measured_cp"]
    # if we were given custom topics
    if args.custom_topics:
        currentTargetTopics = args.custom_topics

    # if we were given an input .bag file
    if args.from_bag:
        # extract data from .bag instead of running listeners
        calibrateSinusRobot = CalibrateRobotTracker(args.output_path, 
                                                    currentTargetTopics)
        calibrateSinusRobot.extractData(args.from_bag)
    else:
        # listen for topics if path to .bag was not provided
        calibrateSinusRobot = CalibrateRobotTracker(args.output_path,
                                                    currentTargetTopics)
        calibrateSinusRobot.runListeners()
