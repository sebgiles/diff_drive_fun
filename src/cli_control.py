#!/bin/env python
import math
import rospy
from geometry_msgs.msg import TwistStamped

curv_mode = False
buffer = {'v':0.0, 'w':0.0, 'k':0.0}

def get_curvature(v, w, tol=0.0):
    if abs(v) <= tol:
        if tol == 0.0:
            return math.nan if w == 0.0 else math.copysign(math.inf, w)
        else:
            v = math.copysign(tol, v)
    return w / v

def send_cmd(e):
    global curv_mode, buffer
    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    v = buffer['v']
    w = v * buffer['k'] if curv_mode else buffer['w']
    msg.twist.linear.x = v
    msg.twist.angular.z = w
    msg.twist.angular.x = get_curvature(v, w)
    pub.publish(msg)


def handle_user_input():
    global curv_mode, buffer
    raw = input().split('=')
    if raw[0] in ["", "exit"]:
        rospy.signal_shutdown("Cli Control terminated by user.")
    if len(raw) != 2:
        print('Invalid Command')
        return
    name, value = raw
    try:
        buffer[name] = float(value)
    except:
        print('Invalid Command')
        return
    if name == 'w':
        curv_mode = False
    if name == 'k':
        if not curv_mode:
            print('Switching to curvature reference')
            curv_mode = True
            if buffer['v'] == 0:
                print("WARNING: v=0")


if __name__ == "__main__":
    rospy.init_node('commander')
    pub = rospy.Publisher('/x/command', TwistStamped, queue_size=1)
    rospy.Timer(rospy.Duration.from_sec(0.05), send_cmd)

    while not rospy.is_shutdown():
        handle_user_input()