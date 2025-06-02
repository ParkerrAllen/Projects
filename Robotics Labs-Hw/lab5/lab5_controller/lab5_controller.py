"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import heapq
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

import sys
sys.path.append('..')
from lab5_joint.lab5_joint import init, calculateIk, moveArmToTarget, checkArmAtPosition, closeGrip, openGrip

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
init(robot, timestep)

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range = robot.getDevice('range-finder')
range.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'
# mode = 'picknplace'



###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    pos = gps.getValues()
    while math.isnan(pos[0]):
        robot.step(timestep)
        pos = gps.getValues()
    start_w = (pos[0], pos[1])
    end_w = (-6.0, -5.0)

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    def w2m(coords):
        wx, wy = coords
        #When multiplied by 30, this makes wx and wy negative and the 360 makes it into a usable index range
        tempx=360+int(wx*30)
        tempy=-int(wy*30)
        #Clamping
        if 0<=tempx<=359:
            x=tempx
        else:
            x=0 if tempx<0 else 359
        if 0<=tempy<=359:
            y=tempy
        else:
            y=0 if tempy<0 else 359
        mx =tempx  
        my =tempy
        #Return the map coordinates
        return (mx, my)
       

    start = w2m(start_w)
    end =w2m(end_w)
    def heuristic(a, b):
        x,y=a
        x1,y1=b
        dx=abs(x1-x)
        dy=abs(y1-y)
        dis=dx+dy
        return dis
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array (360x360) representing the world's configuration space
                    with 0 as free space and 1 as an obstacle.
        :param start: (x, y) tuple representing the start cell in the map.
        :param end: (x, y) tuple representing the end cell in the map.
        :return: A list of (x, y) tuples representing the shortest path from start to end.
                 Returns an empty list if no path is found.
        '''
        sx, sy = start
        ex, ey = end
        nostart=map[sy][sx]==1
        noend=map[ey][ex]==1
        if nostart or noend:
            message = "Start or End is inside an obstacle!"
            print(message)
            return []

        open_set = []
        init_p=0
        init_node=start
        node_data=(init_p,init_node)
        heapq.heappush(open_set, node_data)

        came_from = dict()
        g_score = dict()
        f_score = dict()

        g_score[start] = 0
        hval=heuristic(start,end)
        f_score[start] = hval
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),  
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  

        while len(open_set)>0:
            top=heapq.heappop(open_set)
            cur_prior=top[0]
            cur_node=top[1]
            print("Current Node:",cur_node, "with f score:",cur_prior)
            x,y=cur_node
            if cur_node == end:
                path = []
                node_path=cur_node
                while node_path in came_from:
                    path.append(node_path)
                    parent=came_from[node_path]
                    node_path = parent
                path.append(start)
                reverse=path[::-1]  
                return reverse
               
            for neighbor in neighbors:
                dx=neighbor[0]
                dy=neighbor[1]
                neix=x+dx
                neiy=y+dy
                neighbor=(neix,neiy)
                xbounds=neix>=0 and neix<360
                ybounds=neiy>=0 and neiy<360
                if xbounds and ybounds:
                    cval=map[neiy][neix]
                    if cval == 1:
                        continue
                    h2nei=heuristic(cur_node,neighbor)
                    ten_g_score = g_score[cur_node] + h2nei
                    if neighbor not in g_score :
                        g_score[neighbor]=float('inf')
                    if ten_g_score<g_score[neighbor]:
                        came_from[neighbor]=cur_node
                        g_score[neighbor]=ten_g_score
                        heur2goal=heuristic(neighbor,end)
                        total=ten_g_score+heur2goal
                        f_score[neighbor]=total
                        prior_data=(f_score[neighbor],neighbor)
                        heapq.heappush(open_set,prior_data)
        nopath=True
        if nopath:
            return []
    # Part 2.1: Load map (map.npy) from disk and visualize it
        # plt.imshow(thresholded_map, cmap='gray', origin='upper')  # Change 'lower' to 'upper'
        # plt.colorbar(label="Occupancy (0 = Free, 1 = Obstacle)")
        # plt.title("Thresholded Occupancy Grid Map")
        # plt.show()
    thresholded_map = np.load("map.npy")

    # Part 2.2: Compute an approximation of the “configuration space”
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size))
    expanded_obstacles = convolve2d(thresholded_map, kernel, mode='same', boundary='fill', fillvalue=0)
    config_space = (expanded_obstacles > 0).astype(np.uint8)
    np.save("config_space.npy", config_space)
    # print("Configuration space saved as 'config_space.npy'.")
    # Part 2.3 continuation: Call path_planner
    path = path_planner(config_space, start, end)

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    def m2w(coords):
        mcoordx=coords[0]
        mcoordy=coords[1]
        diff_max_x=359-mcoordx
        sca_diff_x=-(diff_max_x/30.0)
        world_x=sca_diff_x
        sca_diff_y=-(mcoordy/30.0)
        world_y=sca_diff_y
        return (world_x, world_y)
    waypoints = [m2w(pt) for pt in path]
    np.save("path.npy", waypoints)
    print("Converted world path saved as 'path.npy'")

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy", allow_pickle=True)
    # print("Loaded waypoints:", waypoints)
    def w2m(coords):
        wx=coords[0]
        wy=coords[1]
        offsetx=wx+12
        normx=offsetx/12.0
        scalx=normx*359.0
        map_x=int(scalx)
        negy=-wy
        normy=negy/12.0
        scaly=normy*359.0
        map_y=int(scaly)
        #Clamping to make sure they are within the bounds
        if map_x<0:
            x=0
        elif map_x>359:
            x=359
        else:
            x=map_x
        if map_y<0:
            y=0
        elif map_y>359:
            y=359
        else:
            y=map_y
        return (x, y)
    green=int(0x00FF00)
    display.setColor(green)
    for pt in waypoints:
        ap_x, my = w2m(pt)
        dxval=[-1,0,1]
        dyval=[-1,0,1]
        for dx in dxval:
            for dy in dyval:
                ax=ap_x+dx
                ay=my+dy
                xbound=0<=ax<360
                ybound=0<=ay<360
                if xbound and ybound:
                    display.drawPixel(ax,ay)
state = 0 # use this to iterate through your path
goal_index = 0       # Index of the current waypoint
k_rho = 0.75         # Linear gain
k_alpha = 0.5        # Proportional gain for heading error
k_d = -0.0           # Derivative gain for heading error (negative for damping)
wheel_radius = 0.1   # Adjust as needed
prev_alpha = 0.0     # Assumed wheel radius (m) – adjust if needed

def wrap_to_pi(angle):
    x=angle+math.pi
    wa=x%(2*math.pi)
    final=wa-math.pi
    return final

if mode == 'picknplace':
    # Do not change start_ws and end_ws below.
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    # Wait until GPS has a valid reading:
    pos = gps.getValues()
    while math.isnan(pos[0]):
        robot.step(timestep)
        pos = gps.getValues()
    # Define pick and drop locations.
    # (Adjust these coordinates based on your environment.)
    pick_location = (-8.4, -5.7)   # location where object is to be picked up
    drop_location = start_ws[0]      # drop at the start_ws location

    # Re-use the planner functions (world_to_map, map_to_world, heuristic, path_planner)
    # to generate paths from current position -> pick and pick -> drop.
    def w2m(coords):
        wx=coords[0]
        wy=coords[1]
        if math.isnan(wx) or math.isnan(wy):
            print("Invalid world coordinates: wx =", wx, "wy =", wy)
        offsetx=wx+12
        normx=offsetx/12.0
        scalx=normx*359.0
        map_x=int(scalx)
        negy=-wy
        normy=negy/12.0
        scaly=normy*359.0
        map_y=int(scaly)
        #Clamping to make sure they are within the bounds
        if map_x<0:
            x=0
        elif map_x>359:
            x=359
        else:
            x=map_x
        if map_y<0:
            y=0
        elif map_y>359:
            y=359
        else:
            y=map_y
        return (x, y)

    # (Re)load the thresholded map and compute configuration space:
    thresholded_map = np.load("map.npy")
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size))
    expanded_obstacles = convolve2d(thresholded_map, kernel, mode='same', boundary='fill', fillvalue=0)
    config_space = (expanded_obstacles > 0).astype(np.uint8)

    # Get current base position and compute map indices:
    current_base = (gps.getValues()[0], gps.getValues()[1])
    start_map = w2m(current_base)
    pick_map = w2m(pick_location)
    drop_map = w2m(drop_location)

    def heuristic(a, b):
        x,y=a
        x1,y1=b
        dx=abs(x1-x)
        dy=abs(y1-y)
        dis=dx+dy
        return dis

    def path_planner(map_arr, start, end):
        sx, sy = start
        ex, ey = end
        nostart=map[sy][sx]==1
        noend=map[ey][ex]==1
        if nostart or noend:
            message = "Start or End is inside an obstacle!"
            print(message)
            return []

        open_set = []
        init_p=0
        init_node=start
        node_data=(init_p,init_node)
        heapq.heappush(open_set, node_data)

        came_from = dict()
        g_score = dict()
        f_score = dict()

        g_score[start] = 0
        hval=heuristic(start,end)
        f_score[start] = hval
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),  
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  

        while len(open_set)>0:
            top=heapq.heappop(open_set)
            cur_prior=top[0]
            cur_node=top[1]
            print("Current Node:",cur_node, "with f score:",cur_prior)
            x,y=cur_node
            if cur_node == end:
                path = []
                node_path=cur_node
                while node_path in came_from:
                    path.append(node_path)
                    parent=came_from[node_path]
                    node_path = parent
                path.append(start)
                reverse=path[::-1]  
                return reverse
               
            for neighbor in neighbors:
                dx=neighbor[0]
                dy=neighbor[1]
                neix=x+dx
                neiy=y+dy
                neighbor=(neix,neiy)
                xbounds=neix>=0 and neix<360
                ybounds=neiy>=0 and neiy<360
                if xbounds and ybounds:
                    cval=map[neiy][neix]
                    if cval == 1:
                        continue
                    h2nei=heuristic(cur_node,neighbor)
                    ten_g_score = g_score[cur_node] + h2nei
                    if neighbor not in g_score :
                        g_score[neighbor]=float('inf')
                    if ten_g_score<g_score[neighbor]:
                        came_from[neighbor]=cur_node
                        g_score[neighbor]=ten_g_score
                        heur2goal=heuristic(neighbor,end)
                        total=ten_g_score+heur2goal
                        f_score[neighbor]=total
                        prior_data=(f_score[neighbor],neighbor)
                        heapq.heappush(open_set,prior_data)
        nopath=True
        if nopath:
            return []

    def m2w(coords):
        mcoordx=coords[0]
        mcoordy=coords[1]
        diff_max_x=359-mcoordx
        sca_diff_x=-(diff_max_x/30.0)
        world_x=sca_diff_x
        sca_diff_y=-(mcoordy/30.0)
        world_y=sca_diff_y
        return (world_x, world_y)
    # Generate paths:
    pick_path = path_planner(config_space, start_map, pick_map)
    drop_path = path_planner(config_space, pick_map, drop_map)

    pick_waypoints = [m2w(pt) for pt in pick_path]
    drop_waypoints = [m2w(pt) for pt in drop_path]

    pick_index = 0
    drop_index = 0
    state = 0   # 0: navigate to pick, 1: pick, 2: navigate to drop, 3: drop, 4: done

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
   
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            xf=wx*30
            yf=wy*30
            xabs=abs(int(xf))
            yabs=abs(int(yf))
            xflip=360-xabs
            if xflip<0:
                tempx=0
            else:
                tempx=xflip
            if tempx>359:
                x=359
            else:
                x=tempx
            if yabs<0:
                tempy=0
            else:
                tempy=yabs
            if tempy>359:
                y=359
            else:
                y=tempy
            val=map[y][x]
            newval=val+0.005
            if newval>1:
                newval=1
            map[y][x]=newval
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            gray=int(newval*255)
            red=gray<<16
            green=gray<<8
            blue=gray
            color=red+green+blue
            display.setColor(color)
            display.drawPixel(x, y)
            # display.setColor(int(0X0000FF))
            # display.drawPixel(360-abs(int(wx*30)),abs(int(wy*30)))

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            mapc=map.copy()
            tempm=(mapc>=.5)
            threshold=tempm.astype(np.uint8)
            filename="map.npy"
            np.save(filename, threshold)
            print("Thresholded map saved as 'map.npy'")
           
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    elif mode == 'picknplace':
        # -------------------- State Machine for Pick and Place --------------------
        if state == 0:
            # Navigate along pick_waypoints
            if pick_index < len(pick_waypoints):
                target = pick_waypoints[pick_index]
                goalx = target[0]
                goaly = target[1]
                dx = goalx - pose_x
                dy = goaly - pose_y
                sq_diff_x = dx**2
                sq_diff_y = dy**2
                rho = math.sqrt(sq_diff_x + sq_diff_y)
                goal_angle = math.atan2(dy, dx)
                adjusted_pose_theta = wrap_to_pi(pose_theta + 1.2)
                alpha = wrap_to_pi(goal_angle - adjusted_pose_theta)
               
                if rho < 0.1:
                    pick_index += 1
                else:
                    dt = timestep / 1000.0
                    alpha_dot = (alpha - prev_alpha) / dt
                    prev_alpha = alpha
                    angle_threshold = 0.2
                    if abs(alpha) < angle_threshold:
                        fvel = k_rho * rho
                    else:
                        fvel = 0
                    v = fvel
                    w = k_alpha * alpha + k_d * alpha_dot
                    axle_half = AXLE_LENGTH / 2.0
                    left_n = v - (axle_half * w)
                    right_n = v + (axle_half * w)
                    vL_unb = left_n / wheel_radius
                    vR_unb = right_n / wheel_radius

                    if vL_unb < -MAX_SPEED:
                        vL = -MAX_SPEED
                    elif vL_unb > MAX_SPEED:
                        vL = MAX_SPEED
                    else:
                        vL = vL_unb

                    if vR_unb < -MAX_SPEED:
                        vR = -MAX_SPEED
                    elif vR_unb > MAX_SPEED:
                        vR = MAX_SPEED
                    else:
                        vR = vR_unb

                    robot_parts[MOTOR_LEFT].setVelocity(vL)
                    robot_parts[MOTOR_RIGHT].setVelocity(vR)
            else:
                print("Reached pick location.")
                robot_parts[MOTOR_LEFT].setVelocity(0)
                robot_parts[MOTOR_RIGHT].setVelocity(0)
                state = 1


        elif state == 1:
            # At pick location: move arm to pick position.
            px=.685
            py=-.1
            pz=.98
            pick_target = [px, py, pz]  # Adjust these values as needed.
            ik_pick = calculateIk(pick_target)
            moveArmToTarget(ik_pick)
            is_ready=checkArmAtPosition(ik_pick)
            if is_ready:
                closeGrip()
                print("Object grasped.")
                state = 2
            else:
                print("Moving arm to pick position...")

        elif state == 2:
            # Navigate along drop_waypoints.
            if drop_index < len(drop_waypoints):
                target = drop_waypoints[drop_index]
                goalx=target[0]
                goaly=target[1]
                dx =goalx - pose_x
                dy = goaly - pose_y
                sq_diff_x=dx**2
                sq_diff_y=dy**2
                rho = math.sqrt(sq_diff_x + sq_diff_y)
                goal_angle = math.atan2(dy, dx)
                adjusted_pose_theta = wrap_to_pi(pose_theta + 1.2)
                alpha = wrap_to_pi(goal_angle - adjusted_pose_theta)
                if rho < 0.1:
                    drop_index += 1
                else:
                    dt = timestep/1000.0
                    alpha_dot = (alpha - prev_alpha)/dt
                    prev_alpha = alpha
                    angle_threshold = 0.2
                    if abs(alpha) < angle_threshold:
                        fvel = k_rho * rho
                    else:
                        fvel = 0
                    v = fvel
                    w = k_alpha * alpha + k_d * alpha_dot
                    axle_half=AXLE_LENGTH/2.0
                    left_n=v-(axle_half*w)
                    right_n=v+(axle_half*w)
                    vL_unb = left_n/wheel_radius
                    vR_unb = right_n/wheel_radius
                    if vL_unb < -MAX_SPEED:
                        vL = -MAX_SPEED
                    elif vL_unb > MAX_SPEED:
                        vL = MAX_SPEED
                    else:
                        vL = vL_unb
                    if vR_unb < -MAX_SPEED:
                        vR = -MAX_SPEED
                    elif vR_unb > MAX_SPEED:
                        vR = MAX_SPEED
                    else:
                        vR = vR_unb
                    robot_parts[MOTOR_LEFT].setVelocity(vL)
                    robot_parts[MOTOR_RIGHT].setVelocity(vR)
            else:
                print("Reached drop location.")
                robot_parts[MOTOR_LEFT].setVelocity(0)
                robot_parts[MOTOR_RIGHT].setVelocity(0)
                state = 3

        elif state == 3:
            dx=-.3
            dy=-0
            dz=.5
            # At drop location: move arm to drop position.
            drop_target = [dx, dy, dz]  # Adjust as needed.
            ik_drop = calculateIk(drop_target)
            moveArmToTarget(ik_drop)
            if checkArmAtPosition(ik_drop):
                openGrip()
                print("Object released.")
                state = 4
            else:
                print("Moving arm to drop position...")

        elif state == 4:
            # Task complete.
            robot_parts[MOTOR_LEFT].setVelocity(0)
            robot_parts[MOTOR_RIGHT].setVelocity(0)
            print("Pick and Place task complete.")
            break

    else: # not manual mode
        # Part 3.2: Feedback controller
        if len(waypoints) > 0:
            # Get the current goal waypoint (in world coordinates)
            tgt = waypoints[goal_index]
            gx=tgt[0]
            gy=tgt[1]
            # Compute errors
            dx = gx - pose_x
            dy = gy - pose_y
            dx2 = dx ** 2
            dy2 = dy ** 2
            rho = math.sqrt(dx2 + dy2)
           
            offset = 1.2  # Experiment with this value (e.g., -1.0) if needed
            adjusted_pose_theta = pose_theta + offset
           
            goal_angle = math.atan2(dy, dx)
            alpha = wrap_to_pi(goal_angle - adjusted_pose_theta)
           
            # Debug prints
            # compass_vals = compass.getValues()
            # print("Compass raw values:", compass_vals)
            # print(f"Computed pose_theta: {pose_theta:.3f}, Adjusted pose_theta: {adjusted_pose_theta:.3f}")
            # print(f"Current Pose: ({pose_x:.3f}, {pose_y:.3f}, {adjusted_pose_theta:.3f})")
            # print(f"Goal: ({current_goal[0]:.3f}, {current_goal[1]:.3f})")
            # print(f"dx: {dx:.3f}, dy: {dy:.3f}, rho: {rho:.3f}")
            # print(f"Goal angle: {goal_angle:.3f}, alpha: {alpha:.3f}")
           
            # Check if the waypoint is reached
            if rho < 0.1:
                print(f"Reached waypoint {goal_index}")
                if goal_index < len(waypoints) - 1:
                    goal_index += 1
                v = 0
                w = 0
            else:
                dt = timestep / 1000.0  # seconds
                alpha_dot = (alpha-prev_alpha) / dt
                prev_alpha = alpha  # update for the next iteration
               
                # If heading error is large, command zero forward speed to allow turning
                if abs(alpha) > 0.2:
                    v = 0
                else:
                    v = k_rho * rho
                # Combine proportional and derivative terms on alpha
                w = k_alpha * alpha + k_d * alpha_dot
           
            # Differential drive conversion
            axle_half = AXLE_LENGTH / 2.0
            left_n = v - (axle_half * w)
            right_n = v + (axle_half * w)
            vL_raw = left_n / wheel_radius
            vR_raw = right_n / wheel_radius

            if vL_raw < -MAX_SPEED:
                vL = -MAX_SPEED
            elif vL_raw > MAX_SPEED:
                vL = MAX_SPEED
            else:
                vL = vL_raw

            if vR_raw < -MAX_SPEED:
                vR = -MAX_SPEED
            elif vR_raw > MAX_SPEED:
                vR = MAX_SPEED
            else:
                vR = vR_raw
        else:
            vL = 0
            vR = 0

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
   
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
