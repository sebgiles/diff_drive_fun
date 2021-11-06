#!/bin/env python
from dataclasses import dataclass
import math
import numpy as np
import rospy
import tf
from geometry_msgs.msg import TwistStamped, PointStamped
from ullc_diff_drive.msg import DifferentialWheels as Wheels

def clip(x, limit):
    return np.clip(x, -limit, +limit)

def rotate(x, theta):
    assert len(x) == 2
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ x


class System:
    def get_output(self):
        raise NotImplementedError()

    def set_input(self, _u):
        raise NotImplementedError()


@dataclass
class RobotModel:
    d:       float = 0.6  # Distance between wheel contact points
    m:       float = 80   # Mass
    I_CoM:   float = 100  # Moment of inertia about vertical axis through CoM
    I_w:     float = 0.05 # Rotational inertia of wheels
    r:       float = 0.1  # Wheel radius
    l:       float = 0.5  # Forward offset of the CoM from the wheels midpoint
    tau_max: float = 30   # Maximum torque of each actuator
    P_max:   float = 200  # Maximum Power of each actuator
    drag:    float = 2    # Actuator Dynamic friction [Nm/radps]

    def __post_init__(self):
        self.J = np.array([[self.r / 2], [self.r / self.d]]) * [[1, 1], [1, -1]]
        self.J_inv = np.linalg.inv(self.J)
        I_z = self.I_CoM + self.m * self.l ** 2
        B = np.array([[1 / self.m], [self.d / 2 / I_z]]) * [[1, 1], [1, -1]]
        B_inv = np.linalg.inv(B)
        self.M = self.r * B_inv + self.I_w * self.J_inv
        self.M_inv = np.linalg.inv(self.M)
        self.M_prime = self.M @ self.J

    def inverse_kinematics(self, x):
        return self.J_inv @ x
    
    def forward_kinematics(self, h):
        return self.J @ h

    def project_wrench(self, f):
        return self.J.transpose() @ f

    def free_dynamics(self, x):
        drag = 0 # - self.drag * self.inverse_kinematics(x)
        coriolis = np.array([0, - self.m * self.l * x[0] * x[1]])
        free_forces = self.project_wrench(coriolis) + drag
        xdot_free = self.M_inv @ free_forces
        return xdot_free

    def forward_dynamics(self, x, tau):
        return self.M_inv @ tau + self.free_dynamics(x)

    def inverse_dynamics(self, x, xdot):
        return self.M @ (xdot - self.free_dynamics(x))

    def available_torque(self, x):
        h = np.abs(self.inverse_kinematics(x))
        h = np.maximum(h, 0.001)
        tau_P_max = self.P_max / h
        return np.minimum(tau_P_max, self.tau_max)

    def limit_tau(self, x, u):
        u_max = self.available_torque(x)
        return clip(u, u_max)

    def get_curvature(x, tol=0.0):
        w = x[1]
        v = x[0]
        if abs(v) <= tol:
            if tol == 0.0:
                return math.nan if w == 0.0 else math.copysign(math.inf, w)
            else:
                v = math.copysign(tol, v)
        return w / v


class RobotSimulator(System):
    def __init__(self, model: RobotModel=RobotModel(), freq=500.0):
        self._model = model
        self.timestep = 1 / freq
        self._x = np.zeros(2)  # [v, omega]
        self._u = np.zeros(2)  # [tau_R, tau_L]
        self._h = model.inverse_kinematics(self._x)
        self._x_pub = rospy.Publisher('/x', TwistStamped, queue_size=1)
        self._h_pub = rospy.Publisher('/h', Wheels, queue_size=1)
        self._pose_pub = rospy.Publisher('/pose', PointStamped, queue_size=1)
        self._x_msg = TwistStamped()
        self._h_msg = Wheels()
        self._pose_msg = PointStamped()
        self._br = tf.TransformBroadcaster()
        self._position = np.zeros(2)        # self._pose_msg.header.stamp = self._x_msg.header.stamp

        self._orientation = 0.0

    def step(self, _timer_event):
        tau = self._model.limit_tau(self._x, self._u)
        xdot = self._model.forward_dynamics(self._x, tau)
        self._integrate(xdot)
        self._h = self._model.inverse_kinematics(self._x)
        self._x_msg.header.stamp = rospy.Time.now()
        self._x_msg.twist.linear.x = self._x[0]
        self._x_msg.twist.angular.z = self._x[1]
        self._x_msg.twist.angular.x = RobotModel.get_curvature(self._x)
        self._x_pub.publish(self._x_msg)
        self._h_msg.header.stamp = self._x_msg.header.stamp
        self._h_msg.right = self._h[0]
        self._h_msg.left = self._h[1]
        self._h_pub.publish(self._h_msg)
        orientation_3D = tf.transformations.quaternion_from_euler(0, 0, self._orientation)
        position_3D = np.array([*self._position, 0])
        self._pose_msg.point.x = position_3D[0]
        self._pose_msg.point.y = position_3D[1]
        self._pose_msg.header.frame_id = 'world'

        self._pose_pub.publish(self._pose_msg)
        self._br.sendTransform(position_3D, orientation_3D, rospy.Time.now(), 'robot', 'world')

    def _integrate(self, xdot):
        self._x += self.timestep * xdot
        self._orientation += self.timestep * self._x[1]
        v_world = rotate([self._x[0], 0], self._orientation)
        self._position += self.timestep * v_world


    def get_output(self):
        return self._h  # TODO: add sensor model

    def set_input(self, u):
        self._u = u


@dataclass
class ControlParams:
    freq: float = 50.0
    k_FF: float = 10.0
    k_P:  float = 0.0
    k_P2: float = 0.0
    k_I:  float = 0.0


class Controller:
    def __init__(self, model: RobotModel, system: System, params: ControlParams=ControlParams()):
        self._params = params
        self._system = system
        self._model = model
        self._x_des = np.array([0.0, 0.0])
        rospy.Subscriber('/x/command', TwistStamped, callback=self._cmd_callback)
        self._curv_pub = rospy.Publisher('/x/curv', TwistStamped, queue_size=1)
        self._curv_f_pub = rospy.Publisher('/curv/factor', TwistStamped, queue_size=1)
        self._twist_msg = TwistStamped()        
        self._h_pub = rospy.Publisher('/h/curv', Wheels, queue_size=1)
        self._h_cmd_pub = rospy.Publisher('/h/command', Wheels, queue_size=1)
        self._h_dot_pub = rospy.Publisher('/h_dot/ref', Wheels, queue_size=1)
        self._tau_pub = rospy.Publisher('/tau', Wheels, queue_size=1)
        self._tau_des_pub = rospy.Publisher('/tau', Wheels, queue_size=1)
        self._wheel_msg = Wheels()        

    def step(self, _timer_event):
        h = self._system.get_output()
        x = self._model.forward_kinematics(h)
        tau_max = self._model.available_torque(x)
        x_dot_max = self._model.forward_dynamics(x, tau_max)
        h_dot_max = self._model.inverse_kinematics(x_dot_max)
        curv_factor = [Controller.s(x[i], self._x_des[i]) for i in [1, 0]]
        x_curv_des = curv_factor * self._x_des 
        h_des = self._model.inverse_kinematics(x_curv_des)
        h_dot_des = self._params.k_FF * (h_des - h)
        h_dot_lim = clip(h_dot_des, h_dot_max)
        x_dot_lim = self._model.forward_kinematics(h_dot_lim)
        u = self._model.inverse_dynamics(x, x_dot_lim)
        self._system.set_input(u)
        h_cmd = self._model.inverse_kinematics(self._x_des)
        self._set_pub_time(rospy.Time.now())
        self._pub_wheel(self._h_cmd_pub,  h_cmd)
        self._pub_wheel(self._tau_pub,    u)
        self._pub_wheel(self._h_pub,      h_des)
        self._pub_wheel(self._h_dot_pub,  h_dot_lim)
        self._pub_twist(self._curv_pub,   x_curv_des)
        self._pub_twist(self._curv_f_pub, curv_factor)

    def _pub_wheel(self, publisher, h):
        self._wheel_msg.right = h[0]
        self._wheel_msg.left = h[1]
        publisher.publish(self._wheel_msg)

    def _pub_twist(self, publisher, x):
        self._twist_msg.twist.linear.x = x[0]
        self._twist_msg.twist.angular.z = x[1]
        self._twist_msg.twist.angular.x = RobotModel.get_curvature(x)
        publisher.publish(self._twist_msg)

    def _set_pub_time(self, time):
        self._twist_msg.header.stamp = time
        self._wheel_msg.header.stamp = time

    def _cmd_callback(self, msg: TwistStamped):
        self._x_des[0] = msg.twist.linear.x
        self._x_des[1] = msg.twist.angular.z

    def bump(t):
        # This is just a trapezoidal impulse.
        return 2 - np.clip(abs(t), 1, 2)

    def s(y, ydes, tol = 0.01):
        if ydes < 0:
            ydes *= -1  # For simplicity, fix ydes to be positive
            y *= -1  # preserve the sign of the ratio.
        y += (0.2 * (ydes - y))  # Pretend we're closer to target (leak)
        ratio = y / max(tol, ydes)
        weight = min(ydes / tol, 1)
        return weight * ratio + (1 - weight) * Controller.bump(ratio)


def main():
    rospy.init_node("ullc")
    con_model = RobotModel()
    sim_model = RobotModel()
    simulator = RobotSimulator(model=sim_model)
    controller = Controller(model=con_model, system=simulator)
    rospy.Timer(rospy.Duration.from_sec(simulator.timestep), simulator.step)
    rospy.Timer(rospy.Duration.from_sec(1/controller._params.freq), controller.step)
    rospy.on_shutdown(lambda: print("Simulation Terminated."))
    rospy.spin()


if __name__ == "__main__":
    main()
