#!/usr/bin/env python

from __future__ import print_function

import math as m
import copy
import json

import rospy
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF, Robot

from pykdl_utils.kdl_kinematics import KDLKinematics
from hebiros_utils.hebiros_wrapper import HebirosWrapper
from hebiros.msg import CommandMsg, WaypointMsg, TrajectoryGoal

PI = m.pi
NAN = float('nan')


def load_json_file(filename):
    print("Opening {} ...".format(filename))
    json_dict = None
    try:
        with open(filename) as f_in:
            json_dict = json.load(f_in)
    except IOError:
        print("No {} file found.".format(filename))
        pass
    return json_dict


class HebiArmController(object):
    def __init__(self, hebi_group_name, hebi_mapping, hebi_gains, urdf_str, base_link, end_link):
        rospy.loginfo("Creating {} instance".format(self.__class__.__name__))

        self.openmeta_testbench_manifest_path = rospy.get_param('~openmeta/testbench_manifest_path', None)
        if self.openmeta_testbench_manifest_path is not None:
            self._testbench_manifest = load_json_file(self.openmeta_testbench_manifest_path)

            # TestBench Parameters
            self._params = {}
            for tb_param in self._testbench_manifest['Parameters']:
                self._params[tb_param['Name']] = tb_param['Value']  # WARNING: If you use these values - make sure to check the type

            # TestBench Metrics
            self._metrics = {}
            for tb_metric in self._testbench_manifest['Metrics']:  # FIXME: Hmm, this is starting to look a lot like OpenMDAO...
                self._metrics[tb_metric['Name']] = tb_metric['Value']

        if hebi_gains is None:
            hebi_gains = {
                'p': [float(self._params['p_gain'])]*3,
                'i': [float(self._params['i_gain'])]*3,
                'd': [float(self._params['d_gain'])]*3
            }

        hebi_families = []
        hebi_names = []
        for actuator in hebi_mapping:
            family, name = actuator.split('/')
            hebi_families.append(family)
            hebi_names.append(name)
        rospy.loginfo("  hebi_group_name: %s", hebi_group_name)
        rospy.loginfo("  hebi_families: %s", hebi_families)
        rospy.loginfo("  hebi_names: %s", hebi_names)
        self.hebi_wrap = HebirosWrapper(hebi_group_name, hebi_families, hebi_names)
        # Set PID Gains
        cmd_msg = CommandMsg()
        cmd_msg.name = hebi_mapping
        cmd_msg.settings.name = hebi_mapping
        cmd_msg.settings.control_strategy = [4, 4, 4]
        cmd_msg.settings.position_gains.name = hebi_mapping
        cmd_msg.settings.position_gains.kp = hebi_gains['p']
        cmd_msg.settings.position_gains.ki = hebi_gains['i']
        cmd_msg.settings.position_gains.kd = hebi_gains['d']
        cmd_msg.settings.position_gains.i_clamp = [0.25, 0.25, 0.25]  # TODO: Tune me.
        self.hebi_wrap.send_command_with_acknowledgement(cmd_msg)

        if base_link is None or end_link is None:
            robot = Robot.from_xml_string(urdf_str)
            if not base_link:
                base_link = robot.get_root()
                # WARNING: There may be multiple leaf nodes
            if not end_link:
                end_link = [x for x in robot.link_map.keys() if x not in robot.child_map.keys()][0]
        # pykdl
        self.kdl_fk = KDLKinematics(URDF.from_xml_string(urdf_str), base_link, end_link)
        self._active_joints = self.kdl_fk.get_joint_names()

        # joint data
        self.position_fbk = [0.0]*self.hebi_wrap.hebi_count
        self.velocity_fbk = [0.0]*self.hebi_wrap.hebi_count
        self.effort_fbk = [0.0]*self.hebi_wrap.hebi_count

        # joint state publisher
        while not rospy.is_shutdown() and len(self.hebi_wrap.get_joint_positions()) < len(self.hebi_wrap.hebi_mapping):
            rospy.sleep(0.1)
        self._joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=1)
        self.hebi_wrap.add_feedback_callback(self._joint_state_cb)

        # Set up Waypoint/Trajectory objects
        self.start_wp = WaypointMsg()
        self.start_wp.names = self.hebi_wrap.hebi_mapping
        self.end_wp = copy.deepcopy(self.start_wp)
        self.goal = TrajectoryGoal()
        self.goal.waypoints = [self.start_wp, self.end_wp]

        self._hold_positions = [0.0]*3
        self._hold = True
        self.jointstate = JointState()
        self.jointstate.name = self.hebi_wrap.hebi_mapping
        rospy.sleep(1.0)

    def execute(self, rate, joint_states, durations):
        max_height = 0.0

        while not rospy.is_shutdown():
            if len(joint_states) > 0:
                # start waypoint
                self.start_wp.positions = self.position_fbk
                self.start_wp.velocities = [0.0]*self.hebi_wrap.hebi_count
                self.start_wp.accelerations = [0.0]*self.hebi_wrap.hebi_count

                # end waypoint
                self.end_wp.positions = joint_states.pop(0)
                self.end_wp.velocities = [0.0]*self.hebi_wrap.hebi_count
                self.end_wp.accelerations = [0.0]*self.hebi_wrap.hebi_count
                self._hold_positions = self.end_wp.positions

                # action goal
                self.goal.times = [0.0, durations.pop()]

                self._hold = False
                self.hebi_wrap.trajectory_action_client.send_goal(self.goal)
                self.hebi_wrap.trajectory_action_client.wait_for_result()
                self._hold = True

                # check status
                if abs(self.position_fbk[0] - self.end_wp.positions[0]) > 0.0872665:
                    print("Failed to achieve objective position.")
                    break

            else:
                rospy.sleep(2.0)
                fk = self.kdl_fk.forward(self.position_fbk)
                max_height = fk.tolist()[2][3]
                print("Max Height: {}".format(max_height))
                break

            rate.sleep()

        if self.openmeta_testbench_manifest_path is not None:
            steps_completed = 0
            if 7 <= len(joint_states):
                steps_completed = 0
            elif 5 <= len(joint_states) < 7:
                steps_completed = 1
            elif 3 <= len(joint_states) < 5:
                steps_completed = 2
            elif len(joint_states) < 3:
                steps_completed = 3
            self._metrics['steps_completed'] = steps_completed
            self._metrics['max_height'] = max_height
            self._write_metrics_to_tb_manifest()

    def _joint_state_cb(self, msg):
        if not rospy.is_shutdown():
            if self._hold:
                self.jointstate.position = self._hold_positions
                self.jointstate.velocity = []
                self.hebi_wrap.joint_state_publisher.publish(self.jointstate)

            self.position_fbk = self.hebi_wrap.get_joint_positions()
            self.velocity_fbk = self.hebi_wrap.get_joint_velocities()
            self.effort_fbk = self.hebi_wrap.get_joint_efforts()

            jointstate = JointState()
            jointstate.header.stamp = rospy.Time.now()
            jointstate.name = self._active_joints
            jointstate.position = self.position_fbk
            jointstate.velocity = self.velocity_fbk
            jointstate.effort = self.effort_fbk
            self._joint_state_pub.publish(jointstate)

    def _write_metrics_to_tb_manifest(self):
        # Write to testbench_manifest metric
        for tb_metric in self._testbench_manifest['Metrics']:
            if tb_metric['Name'] in self._metrics:
                tb_metric['Value'] = self._metrics[tb_metric['Name']]

        # Save updated testbench_manifest.json
        with open(self.openmeta_testbench_manifest_path, 'w') as savefile:
            json_str = json.dumps(self._testbench_manifest, sort_keys=True, indent=2, separators=(',', ': '))
            savefile.write(json_str)


def main():
    rospy.init_node('arm_controller')
    rate = rospy.Rate(200)

    hebi_group_name = "hebi_3dof_arm"
    hebi_mapping = []

    # fetch ROS Params
    hebi_mapping.append(rospy.get_param('j1_mapping', 'Arm/Base'))
    hebi_mapping.append(rospy.get_param('j2_mapping', 'Arm/Shoulder'))
    hebi_mapping.append(rospy.get_param('j3_mapping', 'Arm/Elbow'))
    urdf_str = rospy.get_param('robot_description', None)
    base_link = rospy.get_param('~base_link', None)
    end_link = rospy.get_param('~end_link', None)
    p_gain = rospy.get_param('~p_gain', None)
    i_gain = rospy.get_param('~i_gain', None)
    d_gain = rospy.get_param('~d_gain', None)

    # create arm controller
    if p_gain is not None and i_gain is not None and d_gain is not None:
        hebi_gains = {
            'p': [p_gain]*3,
            'i': [i_gain]*3,
            'd': [d_gain]*3
        }
    else:
        hebi_gains = None

    hebi_arm_controller = HebiArmController(hebi_group_name=hebi_group_name,
                                            hebi_mapping=hebi_mapping,
                                            hebi_gains=hebi_gains,
                                            urdf_str=urdf_str, base_link=base_link, end_link=end_link)
    joint_states = [[0.0, 0.0, 0.0],
                    [1./4*PI, 0.0872665, 0.0],
                    [3./4*PI, 0.0872665, 0.0],
                    [3./4*PI, 0.0872665, 1./2*PI],
                    [5./4*PI, 0.0872665, 1./2*PI],
                    [5./4*PI, 1./2*PI, 1./2*PI],
                    [7./4*PI, 1./2*PI, 1./2*PI],
                    [2.*PI, 1./2*PI, 1./2*PI],
                    [2.*PI, 0.0, 0.0],
                    [2.*PI, 1./2*PI, 0.0]]
    durations = [2.0]*len(joint_states)  # For now, just do 4 seconds for each step
    hebi_arm_controller.execute(rate, joint_states, durations)


if __name__ == '__main__':
    main()
