#!/usr/bin/env python

import rospy

# import datatypes we want to listen to
from geometry_msgs.msg import PoseStamped

def callback(data): # run when we get a new msg
    rospy.loginfo(rospy.get_caller_id(), data.data)

def listener():

    # init node with name listener
    rospy.init_node('listener', anonymous=True)

    # subscribe to relevant topics
    rospy.Subscriber("", PoseStamped, callback)


    rospy.spin() # keep the node running

if __name__ == '__main__':
    listener()