# lab5_joint.py

from controller import Robot
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import ikpy.utils.plot as plot_utils
import math

# Global variables that will be set by an init() call from the main controller.
robot = None
timestep = None

# Global constants and targets, etc.
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)
CAM_POS = (-0.013797, 0.137, 0.326805)
CAM_WIDTH = 240
CAM_HEIGHT = 135
target_item_list = ["orange"]
vrb = True

# Instead of creating our own Robot instance here,
# we now provide an init function that the main controller must call.
def init(robot_instance, timestep_value):
    """Initialize lab5_joint module with the existing robot instance."""
    global robot, timestep, motors, my_chain
    robot = robot_instance
    timestep = timestep_value

    # Create the chain using the URDF file.
    base_elements = [
        "base_link",
        "base_link_Torso_joint",
        "Torso",
        "torso_lift_joint",
        "torso_lift_link",
        "torso_lift_link_TIAGo front arm_11367_joint",
        "TIAGo front arm_11367"
    ]
    my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=base_elements)
    print("Loaded chain links:")
    print(my_chain.links)

    # Define joint names (same as in your controller)
    global part_names
    part_names = (
        "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
        "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
        "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint"
    )

    # Disable fixed links (or links not needed) in the chain.
    for link_id, link in enumerate(my_chain.links):
        if link.name not in part_names or link.name == "torso_lift_joint":
            print(f"Disabling {link.name} (index: {link_id})")
            my_chain.active_links_mask[link_id] = False

    # Initialize the arm motors and their position sensors.
    motors = []
    for link in my_chain.links:
        if link.name in part_names and link.name != "torso_lift_joint":
            # Use the robot instance passed from the main controller.
            motor = robot.getDevice(link.name)
            if link.name == "torso_lift_joint":
                motor.setVelocity(0.07)
            else:
                motor.setVelocity(1)
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timestep)
            motors.append(motor)

# ----------------- Helper Functions ----------------- #
def rotate_y(x, y, z, theta):
    new_x = x * np.cos(theta) + y * np.sin(theta)
    new_z = z
    new_y = -y * np.sin(theta) + x * np.cos(theta)
    return [-new_x, new_y, new_z]

def lookForTarget(recognized_objects):
    if len(recognized_objects) > 0:
        for item in recognized_objects:
            if any(target_item in str(item.get_model()) for target_item in target_item_list):
                target = recognized_objects[0].get_position()
                dist = abs(target[2])
                if dist < 5:
                    return True
    return False

def checkArmAtPosition(ikResults, cutoff=0.00005):
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
    arm_error = math.sqrt(sum((initial_position[i] - ikResults[i])**2 for i in range(14)))
    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False

def moveArmToTarget(ikResults):
    for res in range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            device = robot.getDevice(my_chain.links[res].name)
            device.setPosition(ikResults[res])
            if vrb:
                print(f"Setting {my_chain.links[res].name} to {ikResults[res]}")

def calculateIk(offset_target, orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
    """
    Calculate the inverse kinematics solution for the chain given a target position (and optionally, orientation).

    Parameters
    ----------
    offset_target : list or array-like of length 3
        A vector specifying the target position [x, y, z] of the end effector in robot coordinates.
    orient : bool, optional
        Whether or not to constrain the end-effector to a desired orientation.
        If False, only the position is considered. Default is True.
    orientation_mode : str, optional
        The axis for orientation constraint. Should be "X", "Y", or "Z". Default is "Y".
    target_orientation : list, optional
        The target orientation vector for the end effector. Default is [0, 0, 1].

    Returns
    -------
    list
        The calculated joint angles from inverse kinematics.
    """
    # Get the number of links in the chain.
    num_links = len(my_chain.links)
    
    # Create an initial_position array with the same number of elements as links.
    initial_position = [0] * num_links

    # Map each motor to its corresponding link index.
    motor_idx = 0
    for i in range(num_links):
        link_name = my_chain.links[i].name
        if link_name in part_names and link_name != "torso_lift_joint":
            if motor_idx < len(motors):
                initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
                motor_idx += 1

    # Optionally clamp the initial guess to the joint bounds for active links.
    for i in range(num_links):
        # Only consider active links that correspond to a motor.
        if my_chain.active_links_mask[i]:
            bounds = my_chain.links[i].bounds  # This is a tuple: (lower, upper)
            # If bounds are defined (non-None), clamp the initial guess.
            if bounds is not None:
                lower, upper = bounds
                if initial_position[i] < lower:
                    print(f"Clamping joint {i} from {initial_position[i]} to lower bound {lower}")
                    initial_position[i] = lower
                elif initial_position[i] > upper:
                    print(f"Clamping joint {i} from {initial_position[i]} to upper bound {upper}")
                    initial_position[i] = upper

    # Debug prints:
    print("=== calculateIk DEBUG ===")
    print("Offset target:", offset_target)
    print("Target orientation:", target_orientation)
    print("Initial joint positions (after clamping):", initial_position)

    # Calculate inverse kinematics.
    if orient:
        ikResults = my_chain.inverse_kinematics(
            offset_target,
            initial_position=initial_position,
            target_orientation=target_orientation,
            orientation_mode=orientation_mode
        )
    else:
        ikResults = my_chain.inverse_kinematics(
            offset_target,
            initial_position=initial_position
        )

    # Compute forward kinematics on the resulting joint angles.
    fk_result = my_chain.forward_kinematics(ikResults)
    end_effector_position = [fk_result[0, 3], fk_result[1, 3], fk_result[2, 3]]
    
    # Calculate error.
    error = math.sqrt(
        (end_effector_position[0] - offset_target[0])**2 +
        (end_effector_position[1] - offset_target[1])**2 +
        (end_effector_position[2] - offset_target[2])**2
    )

    # Debug prints:
    print("IK result joint angles:", ikResults)
    print("Computed end-effector position from FK:", end_effector_position)
    print(f"IK error (Euclidean distance): {error:.6f}")
    print("============================")

    return ikResults


def getTargetFromObject(recognized_objects):
    target = recognized_objects[0].get_position()
    offset_target = [-(target[2]) + 0.22, -target[0] + 0.06, target[1] + 0.97 + 0.2]
    return offset_target

def reachArm(target, previous_target, ikResults, cutoff=0.00005):
    error = 0
    ikTargetCopy = previous_target
    if previous_target is None:
        error = 100
    else:
        error = math.sqrt(sum((target[i] - previous_target[i])**2 for i in range(3)))
    if error > 0.05:
        print(f"Recalculating IK, error too high {error}...")
        ikResults = calculateIk(target)
        ikTargetCopy = target
        moveArmToTarget(ikResults)
    if checkArmAtPosition(ikResults, cutoff=cutoff):
        if vrb:
            print("NOW SWIPING")
        return [True, ikTargetCopy, ikResults]
    else:
        if vrb:
            print("ARM NOT AT POSITION")
    return [False, ikTargetCopy, ikResults]

def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

def openGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

# ------------------ Main Test Block ------------------ #
if __name__ == "__main__":
    # If running this file directly, create a robot instance and run a simple test.
    test_robot = Robot()
    test_timestep = int(test_robot.getBasicTimeStep())
    init(test_robot, test_timestep)
    
    # For testing, move the arm to a preset IK target.
    test_ik = [0,0,0,0,0.07,0,-1.5,2.29,-1.8,1.1,-1.4,0,0,0]
    moveArmToTarget(test_ik)
    
    while test_robot.step(test_timestep) != -1:
        pass
