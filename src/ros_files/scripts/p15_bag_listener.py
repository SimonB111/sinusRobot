#!/usr/bin/env python

import rospy

# import datatypes we want to listen to
from geometry_msgs.msg import PoseStamped

def callback(data: PoseStamped) -> None:
    '''
    invoked when receiving data from /measured_cp,
    updates Robot variables
    Returns: None
    '''
    # position and rotation are float64
    # can access data.pose.position.x (y, z)
    # can access data.pose.orientation.w (x, y, z)
    rospy.loginfo(rospy.get_caller_id() + ' x: %.3f', data.pose.position.x)


def listener() -> None:
    '''
    creates a node that listens to /measured_cp
    Returns: None
    '''
    # init node with name listener
    rospy.init_node('listener', anonymous=True)
    # subscribe to measured_cp topic
    rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, callback)

    rospy.spin() # keep the node running


if __name__ == '__main__':
    listener()