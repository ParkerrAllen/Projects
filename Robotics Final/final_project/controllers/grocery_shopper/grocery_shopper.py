# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
# Std-lib
import math
import os
import re
import pathlib
import random
import heapq
from typing import Optional, Tuple, List, Sequence, Iterable

# Third-party
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# Webots API
from controller import Supervisor, Display, Camera

# ───────────────────────────────────────────────────────────────────────────────
# Robot & Hardware Constants
# ───────────────────────────────────────────────────────────────────────────────
MAX_SPEED        = 7.0          # wheel max [rad/s]
MAX_SPEED_MS     = 0.633        # wheel max [m/s]
AXLE_LENGTH      = 0.4044       # wheel-to-wheel [m]
MOTOR_LEFT       = 10
MOTOR_RIGHT      = 11
N_PARTS          = 12

# LIDAR specs
LIDAR_ANGLE_BINS        = 667
LIDAR_SENSOR_MAX_RANGE  = 5.5   # [m]
LIDAR_SENSOR_ANGLE      = math.radians(240)

# ───────────────────────────────────────────────────────────────────────────────
# Vision / Colour Thresholds
# ───────────────────────────────────────────────────────────────────────────────
yellow_lower = np.array([0, 145, 145])
yellow_upper = np.array([50, 255, 255])

green_lower  = np.array([0, 145,   0])
green_upper  = np.array([50, 255,  50])

MIN_CONTOUR_AREA = 5
MAX_CONTOUR_AREA = 1000

# ───────────────────────────────────────────────────────────────────────────────
# Cube Detection / Tracking
# ───────────────────────────────────────────────────────────────────────────────
cubes               : List[Tuple[float, float, float]] = []
seen_ids            : set[int]  = set()
MATCH_THRESH        = 10         # px, blob→object match tolerance
WORLD_DEDUP_DIST    = 0.10       # m, ignore new cubes closer than this
CAMERA_MOUNT_Z      = 1.05       # m above ground

# ───────────────────────────────────────────────────────────────────────────────
# World Geometry
# ───────────────────────────────────────────────────────────────────────────────
WORLD_X, WORLD_Y    = 28.0, 16.0
HALF_X,  HALF_Y     = WORLD_X / 2, WORLD_Y / 2
ORIG_X,  ORIG_Y     = -HALF_X, -HALF_Y
Point = Tuple[float, float]

GRID_W, GRID_H      = 480, 840
RES                 = GRID_W / WORLD_Y          # cells per metre
CELL_SIZE           = 1.0 / RES

# ───────────────────────────────────────────────────────────────────────────────
# Low-Level Motion Parameters
# ───────────────────────────────────────────────────────────────────────────────
FORWARD_SPEED  = 6.0     # base wheel speed [rad/s]
TURN_SPEED     = 1.0
OBSTACLE_DIST  = 2.5     # [m]

DIAG           = math.radians(60)

# Backup / un-stuck behaviour
BACKUP_TIME_S  = 2.0
backup_steps   = 0
backup_running = False

turning        = False
turn_dir       = 1
turn_steps_left = 0

SIDE_THR        = 0.2
EVADE_REVERSE_S = 0.5
evade           = False
evade_dir       = 1
evade_steps     = 0

# ───────────────────────────────────────────────────────────────────────────────
# Navigation Gains / Thresholds
# ───────────────────────────────────────────────────────────────────────────────
KP_ANG      = 1.0
MAX_ANG     = 1.0
KP_LIN      = 0.8
MAX_LIN     = 0.3
ANG_THRESH  = math.radians(30)

APPROACH_POINT_DIST = 1.0
MAX_CLEARANCE_TEST  = 3.0
CLEARANCE_STEP      = CELL_SIZE
APPROACH_DISTANCE   = 0.3

# ───────────────────────────────────────────────────────────────────────────────
# Configuration-Space Preparation
# ───────────────────────────────────────────────────────────────────────────────
robot_radius_m  = 0.5
inflation_cells = int((robot_radius_m + 0.3) * RES)
kernel          = np.ones((2 * inflation_cells + 1, 2 * inflation_cells + 1),
                          dtype=bool)
cfg_space       = np.zeros((GRID_H, GRID_W), dtype=int)

# ───────────────────────────────────────────────────────────────────────────────
# Navigation State
# ───────────────────────────────────────────────────────────────────────────────
current_target : Optional[Tuple[float, float]] = None
visited        : set[int] = set()

# ───────────────────────────────────────────────────────────────────────────────
# Webots Robot Setup
# ───────────────────────────────────────────────────────────────────────────────
robot     = Supervisor()
timestep  = int(robot.getBasicTimeStep())

print("Starting")
# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)


# ----------  IKPY  SET-UP  ----------
URDF_PATH = pathlib.Path("tiago_generated.urdf")

print("Exporting & patching URDF …")
raw_urdf = robot.getUrdf()
raw_urdf = re.sub(r'type\s*=\s*"(?:continuous|floating|planar)"',
                  'type="revolute"', raw_urdf)
URDF_PATH.write_text(raw_urdf)

# build a chain that starts at the torso plate (arm root)

arm_chain = Chain.from_urdf_file(
    str(URDF_PATH),
    base_elements=["arm_1_joint"],          # ← start IK at the first arm joint
    base_element_type="joint"
)

# grab every Webots device whose key starts with "arm_" and still ends with "_joint"
ARM_JOINT_NAMES = [n for n in robot_parts.keys()
                   if n.startswith("arm_") and n.endswith("_joint")][:7]
if len(ARM_JOINT_NAMES) != 7:
    raise RuntimeError(f"Expected 7 arm joints, found {ARM_JOINT_NAMES}")

# map them into the IKPy chain -----------------------------
name_to_idx = {l.name: i for i, l in enumerate(arm_chain.links)}

arm_indices, mask, arm_motors = [], [0]*len(arm_chain.links), []
for jname in ARM_JOINT_NAMES:
    if jname not in name_to_idx:          # still mismatch? dump names
        raise RuntimeError(
            f"Joint {jname} not in IK chain.\n"
            f"Links seen: {[l.name for l in arm_chain.links]}")
    idx = name_to_idx[jname]
    arm_indices.append(idx)
    mask[idx] = 1
    arm_motors.append(robot_parts[jname])
for m in arm_motors:
    ps = m.getPositionSensor()
    if ps:                           # every motor has one
        ps.enable(timestep)

active_links_mask = [False] * len(arm_chain.links)
for idx in arm_indices:          # only the 7 arm joints move
    active_links_mask[idx] = True
active_links_mask[-1] = False

# Odometry
robot_x = 0
robot_y = 0
robot_yaw = 0

left_vel = 0
right_vel = 0

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
torso_sensor = robot.getDevice("torso_lift_joint_sensor")
torso_sensor.enable(timestep)
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable LiDAR
lidar_dev = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar_dev.enable(timestep)
lidar_dev.enablePointCloud()

# Enable display
display = robot.getDevice("display")
disp_w, disp_h = display.getWidth(), display.getHeight()

camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_SENSOR_ANGLE/2., +LIDAR_SENSOR_ANGLE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar_dev readings not blocked by robot chassis

left_wheel = robot_parts["wheel_left_joint"]
right_wheel = robot_parts["wheel_right_joint"]

left_wheel.setVelocity(0.0)
right_wheel.setVelocity(0.0)

manual_mode = False
gripper_status="closed"

current_cube_idx = None
current_path = []
waypoint_index = 0
visited = set()

map = None

STUCK_DIST = 0.1
STUCK_TIME_S = 3.0
stuck_steps = 0
stuck_threshold = int((STUCK_TIME_S * 1000.0) / timestep)
last_pose = (robot_x, robot_y)

# RRT* Class
class RRTStar:
    class Node:
        def __init__(self, x, y, parent=None, cost=0.0):
            self.x      = x
            self.y      = y
            self.parent = parent
            self.cost   = cost

    def __init__(self, cfg_space, res, world_x, world_y):
        self.config    = cfg_space   # binary mask: 1=obstacle, 0=free
        self.res       = res            # metres per grid cell
        self.world_x   = world_x
        self.world_y   = world_y
        self.nodes     = []

    def sample(self):
        return (random.uniform(-self.world_x/2, self.world_x/2),
                random.uniform(-self.world_y/2, self.world_y/2))

    def in_collision(self, x, y):
        row, col = world_to_grid(x, y)
        return self.config[row, col] == 1

    def segment_collision(self, x1, y1, x2, y2):
        dist  = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(dist / (self.res * 0.5)))
        for i in range(steps + 1):
            t  = i / steps
            xi = x1 + (x2 - x1) * t
            yi = y1 + (y2 - y1) * t
            if self.in_collision(xi, yi):
                return True
        return False

    def nearest(self, x, y):
        best, bd = None, float('inf')
        for n in self.nodes:
            d = (n.x - x)**2 + (n.y - y)**2
            if d < bd:
                bd, best = d, n
        return best

    def steer(self, from_node, to_point, step_size=0.2):
        dx, dy = to_point[0] - from_node.x, to_point[1] - from_node.y
        dist = math.hypot(dx, dy)
        if dist <= step_size:
            return to_point
        return (from_node.x + dx/dist*step_size,
                from_node.y + dy/dist*step_size)

    def path_to_root(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def solve(self, start, goal, max_iters=5000, step=0.2, goal_rad=0.3):
        self.nodes = [self.Node(*start)]
        for i in range(max_iters):
            # bias sampling toward goal 10% of the time
            q_rand = goal if random.random() < 0.1 else self.sample()
            q_near = self.nearest(*q_rand)
            q_new_pt = self.steer(q_near, q_rand, step)
            if self.in_collision(*q_new_pt):
                continue
            # skip if edge crosses an obstacle
            if self.segment_collision(q_near.x, q_near.y, *q_new_pt):
                continue
            # create new node
            new_cost = q_near.cost + math.hypot(q_new_pt[0] - q_near.x,
                                                q_new_pt[1] - q_near.y)
            q_new = self.Node(q_new_pt[0], q_new_pt[1], parent=q_near, cost=new_cost)

            # rewire neighbors within radius
            radius = step * 2.0
            for n in list(self.nodes):
                if math.hypot(n.x - q_new.x, n.y - q_new.y) < radius:
                    # cost to come via q_new
                    via_cost = q_new.cost + math.hypot(n.x - q_new.x, n.y - q_new.y)
                    if via_cost < n.cost and not self.segment_collision(n.x, n.y, q_new.x, q_new.y):
                        n.parent = q_new
                        n.cost   = via_cost

            self.nodes.append(q_new)

            # check if goal reached
            if math.hypot(q_new.x - goal[0], q_new.y - goal[1]) < goal_rad:
                print(f"RRT* reached goal, {i} iterations")
                return self.path_to_root(q_new)

        print("RRT* failed")
        return None

# Helper Functions

def full_joint_vector(q_arm):
    full = [0.0] * len(arm_chain.links)
    for idx, q in zip(arm_indices, q_arm):
        full[idx] = q
    return full

def open_gripper():
    robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    robot_parts["gripper_right_finger_joint"].setPosition(0.045)
    if left_gripper_enc.getValue()>=0.044: 
        gripper_status="open"
                
def close_gripper():
    robot_parts["gripper_left_finger_joint"].setPosition(0)
    robot_parts["gripper_right_finger_joint"].setPosition(0)
    if right_gripper_enc.getValue()<=0.005:
        gripper_status="closed"

def wait(seconds):
    steps = int((seconds * 1000.0) / timestep)
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break

def world_to_grid(wx, wy):
    row = int((HALF_X - wx) * RES)
    col = int((HALF_Y - wy) * RES)
    row = max(0, min(GRID_H-1, row))
    col = max(0, min(GRID_W-1, col))
    return row, col

def find_approach_point(
    wx: float,
    wy: float,
    cfg_space: np.ndarray
) -> Optional[Tuple[float, float]]:
    best_direction: Optional[int] = None
    min_distance: float = MAX_CLEARANCE_TEST + CLEARANCE_STEP  # sentinel > max

    for direction in (1, -1):                # 1 → +y,  -1 → –y
        distance = 0.0
        while distance <= MAX_CLEARANCE_TEST:
            row, col = world_to_grid(wx, wy + direction * distance)

            # Found free cell ─ stop scanning this direction
            if cfg_space[row, col] == 0:
                if distance < min_distance:
                    min_distance = distance
                    best_direction = direction
                break
            distance += CLEARANCE_STEP

    if best_direction is None:
        return None  # No approach point in allowed clearance

    # Step to a safe offset before returning
    return wx, wy + best_direction * APPROACH_POINT_DIST

def smooth_path(
    raw_path: Sequence[Point],
    planner,
    iters: int = 200,
    rng: random.Random | None = None
) -> List[Point]:
    rng = rng or random
    if len(raw_path) < 3:
        return list(raw_path)

    path: List[Point] = list(raw_path)       # make a writable copy
    for _ in range(iters):
        if len(path) < 3:                    # nothing left to prune
            break

        i = rng.randint(0, len(path) - 2)
        j = rng.randint(i + 1, len(path) - 1)

        x1, y1 = path[i]
        x2, y2 = path[j]

        # keep the segment only if it’s collision‑free
        if not planner.segment_collision(x1, y1, x2, y2):
            del path[i + 1 : j]              # remove interior points
    return path

def make_seed(chain, q_active):
    full = [0.0] * len(chain.links)
    for idx, q in zip(arm_indices, q_active):
        full[idx] = q

    eps = 1e-4
    for i, link in enumerate(chain.links):
        if link.bounds is None:
            # IKPy leaves bounds None for virtual links; nothing to clamp
            continue
        lo, hi = link.bounds
        # tighten extremely narrow bounds to avoid lo == hi
        if hi - lo < 1e-6:
            lo -= 1e-6
            hi += 1e-6
        full[i] = max(lo + eps, min(hi - eps, full[i]))
    return full

def reached(pt, tol=0.1):
    return math.hypot(robot_x-pt[0], robot_y-pt[1]) < tol

# -------------------------------------------------------------------

# Loading occupancy grid if it exists
map_path = "map.npy"
if os.path.exists(map_path):
    # load the thresholded 0/1 map and convert to floats 0.0–1.0
    loaded = np.load(map_path)
    occ_grid = loaded.astype(float)

    print(f"Loaded existing map")
    obstacle_mask = (occ_grid >= 0.5)
    config_space_mask = binary_dilation(obstacle_mask, structure=kernel)
    cfg_space = config_space_mask.astype(int)

    for r in range(GRID_H):
        for c in range(GRID_W):
            val = cfg_space[r, c]  # should be between 0.0 and 1.0
    
            # Blue to Red gradient
            r_col = int(val * 255)
            g_col = 0
            b_col = int((1.0 - val) * 255)
    
            color = (r_col << 16) | (g_col << 8) | b_col
            display.setColor(color)
            display.drawPixel(c, r)
else:
    # fresh start
    occ_grid = np.zeros((GRID_H, GRID_W), dtype=float)
    print(f"Initialized empty grid {GRID_H}×{GRID_W}")

# Loading saved cubes
cubes_path = "cubes.npy"
if os.path.exists(cubes_path):
    # load the thresholded 0/1 map and convert to floats 0.0–1.0
    loaded = np.load(cubes_path, allow_pickle=True)
    cubes = [tuple(v) for v in loaded.tolist()]

    print(f"Loaded {len(cubes)} saved cube positions")

    for cube in cubes:
        wx, wy, wz = cube
        rr, cc = world_to_grid(wx, wy)
        display.setColor(0x0000FF)
        display.fillRectangle(cc-2, rr-2, 6, 6)

else:
    cubes = []

# STARTING
# Either map or plan
step = "plan"  

planner = RRTStar(cfg_space, CELL_SIZE, WORLD_X, WORLD_Y)

# Main Loop
# -------------------------------------------------------------------
while robot.step(timestep) != -1:

    # ------------------------------------------------------------------
    robot_x = gps.getValues()[0] # north +x, south -x
    robot_y = gps.getValues()[1] # west +y, east -y   
    n = compass.getValues()
    robot_yaw = math.atan2(-n[0], -n[2])
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    
    key = keyboard.getKey()
    ARM_STEP = 0.02   # metres per press

    if key in (ord('I'),ord('K'),ord('J'),ord('L'),ord('U'),ord('O')):
        # read current IK target (we store it in globals the first time)
        try:
            x_t, y_t, z_t
        except NameError:
            x_t, y_t, z_t = 0.50, 0.0, 0.90
    
        if   key == ord('I'): 
            x_t += ARM_STEP 
            print("I")
        elif key == ord('K'): 
            x_t -= ARM_STEP 
            print("K")
        elif key == ord('J'): y_t += ARM_STEP
        elif key == ord('L'): y_t -= ARM_STEP
        elif key == ord('U'): z_t += ARM_STEP
        elif key == ord('O'): z_t -= ARM_STEP
    
        # quick clamps
        x_t = max(-HALF_X, min(HALF_X, x_t))
        y_t = max(-HALF_Y, min(HALF_Y, y_t))
        z_t = max(0.60,   min(1.25,  z_t))
    
        # solve IK
        q_arm_now = [m.getPositionSensor().getValue() for m in arm_motors]
        seed = make_seed(arm_chain, q_arm_now)
        sol = arm_chain.inverse_kinematics(             # ← remove active_links_mask
                target_position=[x_t, y_t, z_t],
                initial_position=seed)
                
    
        # apply only the 7 arm values (indices ­10..-4 in this mask)
        for angle, motor in zip(sol[-10:-4], arm_motors):
            motor.setPosition(float(angle))
            
    # ── KEYBOARD CONTROL ─────────────────────────────────────────────────────
    if key == ord('S'):
        manual_mode = not manual_mode
        print("MANUAL control" if manual_mode else "AUTO control")

    if manual_mode:
        # arrow keys map to wheel velocities
        if   key == keyboard.UP:
            l_cmd = r_cmd =  FORWARD_SPEED
        elif key == keyboard.DOWN:
            l_cmd = r_cmd = -FORWARD_SPEED
        elif key == keyboard.LEFT:
            l_cmd, r_cmd = -TURN_SPEED,  TURN_SPEED
        elif key == keyboard.RIGHT:
            l_cmd, r_cmd =  TURN_SPEED, -TURN_SPEED
        else:
            l_cmd = r_cmd = 0.0

        left_vel, right_vel = l_cmd, r_cmd

    if key == ord('D'):
        # ── SAVE CURRENT MAP + CUBE LIST ─────────────────────────────────────
        snapshot = (occ_grid > 0.5).astype(int)
        np.save("map.npy",  snapshot)
        np.save("cubes.npy", cubes)
        print("Map and cube coords saved")
        
    if key == ord('R') and step == "grab":
        print("[MANUAL] Forcing transition → reverse")
        close_gripper()           # make sure the fingers are shut
        wait(0.5)                 # tiny pause so the joint settles
        step = "reverse"
        continue                  # jump to next loop iteration
    
    # DETECTION
    # ------------------------------------------------------------------
    if step == "detect":
        # ── VISION-BASED CUBE DISCOVERY ──────────────────────────────────────
        torso_z         = torso_sensor.getValue()
        cam_z           = torso_z + CAMERA_MOUNT_Z
        # print(f"Torso={torso_z:.3f} m   Camera={cam_z:.3f} m")

        # ── 1.  Acquire and threshold the camera image ──────────────────────
        frame_w, frame_h = camera.getWidth(), camera.getHeight()
        raw_bytes        = camera.getImage()                    # BGRA ordering
        img_bgra         = np.frombuffer(raw_bytes, np.uint8).reshape((frame_h, frame_w, 4))
        img_bgr          = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        yellow_mask = cv2.inRange(img_bgr, yellow_lower, yellow_upper)
        kernel      = np.ones((2, 2), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        recognized  = camera.getRecognitionObjects()

        # ── 2.  Iterate over candidate blobs ─────────────────────────────────
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

            # ── 3.  Match blob to a Webots recognition object ───────────────
            best_match      = None
            best_pixel_dist = MATCH_THRESH
            for obj in recognized:
                px, py = obj.getPositionOnImage()
                d      = math.hypot(px - cx, py - cy)
                if d < best_pixel_dist:
                    best_pixel_dist = d
                    best_match      = obj
            if best_match is None:
                continue

            # ── 4.  Retrieve cube’s world coordinates via node ID ───────────
            cube_node   = robot.getFromId(best_match.getId())
            world_pos   = cube_node.getField("translation").getSFVec3f()
            wx, wy, wz  = world_pos

            # discard if already logged (within WORLD_DEDUP_DIST)
            if any(math.hypot(wx - x0, wy - y0) < WORLD_DEDUP_DIST for x0, y0, _ in cubes):
                continue

            # ── 5.  Store and visualise ─────────────────────────────────────
            cubes.append((wx, wy, wz))

            r, c = world_to_grid(wx, wy)
            display.setColor(0x00FFFF)
            display.fillRectangle(c - 2, r - 2, 6, 6)

            print(f"Cube @ ({wx:.2f}, {wy:.2f}, {wz:.2f})")
            np.save("cubes.npy", cubes)

        step = "map"
    # ------------------------------------------------------------------

    elif step == "plan":
        # ── PICK THE NEXT TARGET CUBE ────────────────────────────────────────
        pending = [idx for idx in range(len(cubes)) if idx not in visited]
        if not pending:
            left_vel = right_vel = 0.0
            continue

        # order candidates by straight-line distance from the robot
        pending.sort(key=lambda idx: math.hypot(cubes[idx][0] - robot_x,
                                                cubes[idx][1] - robot_y))

        current_cube_idx = pending[0]
        goal_x, goal_y, goal_z = cubes[current_cube_idx]
        print(f"Heading to cube {current_cube_idx} at ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f})")

        # ── SELECT AN APPROACH POINT NEAR THE CUBE ───────────────────────────
        approach_pt = find_approach_point(goal_x, goal_y, cfg_space)
        if approach_pt is None:
            print(f"  × No approach found for cube {current_cube_idx}")
            visited.add(current_cube_idx)
            step = "plan"
            continue

        ax, ay = approach_pt

        # ── PLAN A PATH (RRT*) ───────────────────────────────────────────────
        raw_path = planner.solve((robot_x, robot_y), (ax, ay))
        if raw_path is None:
            print(f"  × Planner failed for cube {current_cube_idx}")
            visited.add(current_cube_idx)
            step = "plan"
            continue

        refined_path = smooth_path(raw_path, planner, iters=150) or raw_path
        refined_path[-1] = (ax, ay)                 # ensure final node = goal
        print(f"   raw: {len(raw_path)}, smoothed: {len(refined_path)}")

        current_path   = refined_path
        waypoint_index = 0

        # ── VISUALISE THE PATH ───────────────────────────────────────────────
        display.setColor(0xFF00FF)
        # draw segments
        for (x1, y1), (x2, y2) in zip(current_path[:-1], current_path[1:]):
            r1, c1 = world_to_grid(x1, y1)
            r2, c2 = world_to_grid(x2, y2)
            display.drawLine(c1, r1, c2, r2)
        # draw each waypoint
        for px, py in current_path:
            pr, pc = world_to_grid(px, py)
            display.fillRectangle(pc - 2, pr - 2, 5, 5)

        step = "follow_path"
        continue
        
    # NAVIGATION
    
    # ------------------------------------------------------------------
    elif step == "follow_path":
        # ── FOLLOW THE CURRENT PATH ───────────────────────────────────────────
        if waypoint_index < len(current_path):
            # ── 1.  Geometry to the active waypoint ──────────────────────────
            tgt_x, tgt_y          = current_path[waypoint_index]
            delta_x, delta_y      = tgt_x - robot_x, tgt_y - robot_y
            straight_line_dist    = math.hypot(delta_x, delta_y)

            desired_heading       = math.atan2(delta_x, delta_y) - math.pi / 2
            yaw_err_raw           = desired_heading - robot_yaw
            yaw_err               = math.atan2(math.sin(yaw_err_raw),
                                               math.cos(yaw_err_raw))

            # ── 2.  Controller: first angular, then linear ───────────────────
            w_cmd = max(-MAX_ANG,
                        min(MAX_ANG, KP_ANG * yaw_err))

            if abs(yaw_err) < ANG_THRESH:
                v_cmd = min(MAX_LIN, KP_LIN * straight_line_dist)
            else:
                v_cmd = 0.0      # pause forward motion until we face target

            # secondary check: don’t roll if we’re way off‐axis
            if abs(yaw_err) >= math.radians(20):
                v_cmd = 0.0

            # ── 3.  Convert to wheel speeds ──────────────────────────────────
            left_vel  =  (v_cmd + (AXLE_LENGTH / 2) * w_cmd) * (MAX_SPEED / MAX_SPEED_MS)
            right_vel =  (v_cmd - (AXLE_LENGTH / 2) * w_cmd) * (MAX_SPEED / MAX_SPEED_MS)

            # clip to hardware limits
            left_vel  = max(-MAX_SPEED, min(MAX_SPEED, left_vel))
            right_vel = max(-MAX_SPEED, min(MAX_SPEED, right_vel))

            # ── 4.  Waypoint reached? ────────────────────────────────────────
            if straight_line_dist < 0.15:
                waypoint_index += 1
                print(f"Reached waypoint {waypoint_index}/{len(current_path)}")

        else:
            step = "turn_cube"


    # APPROACH CUBE
    # ------------------------------------------------------------------
    elif step == "turn_cube":
        # ── ORIENT THE BASE TOWARD THE SELECTED CUBE ────────────────────────
        cube_x, cube_y, _ = cubes[current_cube_idx]

        dx = cube_x - robot_x
        dy = cube_y - robot_y

        desired_yaw = math.atan2(dx, dy) - math.pi / 2
        yaw_err     = math.atan2(
            math.sin(desired_yaw - robot_yaw),
            math.cos(desired_yaw - robot_yaw),
        )

        if abs(yaw_err) > 0.05:
            spin_dir  = 1.0 if yaw_err > 0 else -1.0
            v_left  =  spin_dir * TURN_SPEED
            v_right = -spin_dir * TURN_SPEED
        else:
            v_left = v_right = 0.0
            robot_parts["wheel_left_joint"].setVelocity(v_left)
            robot_parts["wheel_right_joint"].setVelocity(v_right)
            step = "set_robot_height"
    
    elif step == "set_robot_height":
        # ── ADJUST TORSO TO MATCH CUBE LEVEL ─────────────────────────────────
        torso_now = torso_sensor.getValue()
        cube_z    = cubes[current_cube_idx][2]

        print(f"[SET_HEIGHT] cube={current_cube_idx}  torso={torso_now:.2f} m")

        # choose target: low cubes (< 0.9 m) → floor level, otherwise raise
        desired = 0.0 if cube_z < 0.9 else 0.35

        # move only if we’re not already close enough
        if abs(torso_now - desired) > 0.01:
            robot_parts["torso_lift_joint"].setPosition(desired)
            wait(8.0)   # give the lift time to settle

        step = "grab"


    # GRAB CUBE
    elif step == "grab":
        wx, wy, wz = cubes[current_cube_idx]
        grasp = [wx - 0.10, wy, wz]
        
        q_current = [m.getPositionSensor().getValue() for m in arm_motors]
        seed = make_seed(arm_chain, q_current)
        sol  = arm_chain.inverse_kinematics(             # ← remove active_links_mask
                target_position=grasp,
                initial_position=seed)
        # send only arm joints
        for idx, motor in zip(arm_indices, arm_motors):
            motor.setPosition(float(sol[idx]))
    
        # simple arrival check
        if max(abs(m.getTargetPosition() - m.getPositionSensor().getValue())
               for m in arm_motors) < 0.03:
            close_gripper()
            wait(0.5)
            step = "reverse"
        else:
            left_vel = right_vel = 0.0
        
        
        
        # REVERSE FROM CUBE
    elif step == "reverse":
        # ── BACK AWAY FROM THE CUBE ───────────────────────────────────────────
        back_speed = -0.5 * FORWARD_SPEED
        left_wheel, right_wheel = (
            robot_parts["wheel_left_joint"],
            robot_parts["wheel_right_joint"],
        )

        # start rolling back
        for wheel in (left_wheel, right_wheel):
            wheel.setVelocity(back_speed)

        wait(2.0)   # let the robot clear the cube

        # hit the brakes
        for wheel in (left_wheel, right_wheel):
            wheel.setVelocity(0.0)

        wait(0.5)   # brief pause

        visited.add(current_cube_idx)   # cube handled
        step = "plan"
        continue
        
    # MAPPING
    # ------------------------------------------------------------------
    elif step == "map":

        # Trim the raw scan to exclude the robot’s own body slice
        full_scan      = lidar_dev.getRangeImage()
        useful_reading = full_scan[83 : -83]

        # ── UPDATE OCCUPANCY GRID ───────────────────────────────────────────────
        for idx, distance in enumerate(useful_reading):

            # Ignore returns beyond the sensor’s published max
            if distance > LIDAR_SENSOR_MAX_RANGE:
                continue

            angle   = lidar_offsets[idx]          # beam bearing in robot frame
            lx      = math.cos(angle) * distance  # local X (robot frame)
            ly      = math.sin(angle) * distance  # local Y (robot frame)

            # Convert to world coordinates
            wx = robot_x +  math.cos(robot_yaw) * lx - math.sin(robot_yaw) * ly
            wy = robot_y + -math.sin(robot_yaw) * lx - math.cos(robot_yaw) * ly

            # Clamp to arena boundaries (leave 0.1 m margin)
            wx = max(-HALF_X + 0.1, min(wx,  HALF_X - 0.1))
            wy = max(-HALF_Y + 0.1, min(wy,  HALF_Y - 0.1))

            r, c = world_to_grid(wx, wy)
            occ_grid[r, c] = min(occ_grid[r, c] + 0.005, 1.0)

            display.setColor(0xFFFFFF)            # white pixel
            display.drawPixel(c, r)

        # ── DRAW ROBOT’S CELL ───────────────────────────────────────────────────
        rob_r, rob_c = world_to_grid(robot_x, robot_y)
        display.setColor(0xFF0000)
        display.fillRectangle(rob_c - 1, rob_r - 1, 2, 2)

        # ── STUCK-DETECTION LOGIC ───────────────────────────────────────────────
        cur_pose_x, cur_pose_y = robot_x, robot_y
        move_delta = math.hypot(cur_pose_x - last_pose[0],
                                cur_pose_y - last_pose[1])

        stuck_steps = stuck_steps + 1 if move_delta < STUCK_DIST else 0
        last_pose   = (cur_pose_x, cur_pose_y)

        if stuck_steps >= stuck_threshold:
            if not backup_running:
                backup_running = True
                backup_steps   = int(BACKUP_TIME_S * 1000 / timestep)

            if backup_steps > 0:                         # reverse
                backup_steps -= 1
                left_vel = right_vel = -FORWARD_SPEED
            else:                                        # finished backing up → spin
                backup_running  = False
                stuck_steps     = 0
                turning         = True
                turn_dir        = random.choice([-1, 1])
                turn_steps_left = int(random.uniform(0.5, 1.5) * 1000 / timestep)
                left_vel  = -turn_dir * TURN_SPEED
                right_vel =  turn_dir * TURN_SPEED

        # ── OBSTACLE CHECK ON THREE KEY LIDAR BEAMS ────────────────────────────
        raw_scan   = np.array(lidar_dev.getRangeImage())
        trimmed    = raw_scan[83 : raw_scan.size - 83]

        idx_center = np.argmin(np.abs(lidar_offsets - 0.0))
        idx_left   = np.argmin(np.abs(lidar_offsets -  DIAG))
        idx_right  = np.argmin(np.abs(lidar_offsets +  DIAG))

        d_center = trimmed[idx_center]
        d_left   = trimmed[idx_left]
        d_right  = trimmed[idx_right]

        # Treat out-of-range as ∞ for comparison simplicity
        d_center = float('inf') if d_center > LIDAR_SENSOR_MAX_RANGE else d_center
        d_left   = float('inf') if d_left   > LIDAR_SENSOR_MAX_RANGE else d_left
        d_right  = float('inf') if d_right  > LIDAR_SENSOR_MAX_RANGE else d_right

        forward_blocked = (
            d_center < OBSTACLE_DIST or
            d_left   < OBSTACLE_DIST / 5 or
            d_right  < OBSTACLE_DIST / 5
        )

        # ── BUMP-AND-TURN BEHAVIOR ──────────────────────────────────────────────
        if not turning:
            if not forward_blocked:
                left_vel = right_vel = FORWARD_SPEED
            else:
                # Choose the clearer side (heuristic)
                if d_left > OBSTACLE_DIST / 4 and d_right <= OBSTACLE_DIST / 4:
                    turn_dir = -1
                elif d_right > OBSTACLE_DIST / 4 and d_left <= OBSTACLE_DIST / 4:
                    turn_dir = +1
                elif d_left > d_right:
                    turn_dir = -1
                else:
                    turn_dir = +1

                turning         = True
                turn_steps_left = int(random.uniform(0.5, 1.5) * 1000 / timestep)
                left_vel  = -turn_dir * TURN_SPEED
                right_vel =  turn_dir * TURN_SPEED
        else:
            if turn_steps_left > 0:
                turn_steps_left -= 1
                left_vel  = -turn_dir * TURN_SPEED
                right_vel =  turn_dir * TURN_SPEED
            else:
                turning  = False
                left_vel = right_vel = FORWARD_SPEED

        # ── SAVE A THRESHOLDED SNAPSHOT OF THE MAP (DEBUG) ─────────────────────
        np.save("map.npy", (occ_grid > 0.5).astype(int))

        # Move to next FSM state
        step = "detect"
        continue

        # ------------------------------------------------------------------

    else:
        pass
    
    robot_parts["wheel_left_joint"].setVelocity(left_vel)
    robot_parts["wheel_right_joint"].setVelocity(right_vel)
