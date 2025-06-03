# Copyright (2025) University of Colorado Boulder
# CSCI 3302: Introduction to Robotics

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
import numpy as np
from controller import Robot, Motor, DistanceSensor

# ------------------ CONFIGURATION ------------------ #
LIDAR_SENSOR_MAX_RANGE = 3  # Meters
LIDAR_ANGLE_BINS = 21  # Number of LIDAR angle bins
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees in radians
EPUCK_AXLE_DIAMETER = 0.053  # Distance between wheels (m)
MAX_SPEED = 6.28  # Maximum wheel speed

# These are your pose values that you will update by solving the odometry equations
pose_x, pose_y, pose_theta = 0.197, 0.678, -np.pi

# ------------------ INITIALIZATION ------------------ #
robot = Robot()
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor, rightMotor = robot.getDevice('left wheel motor'), robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)
gsr = [0, 0, 0]

# Initialize the Display
display = robot.getDevice("display")

# get and enable lidar 
lidar = robot.getDevice("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

##### DO NOT MODIFY ANY CODE ABOVE THIS #####

##### Part 1: Setup Data structures #####
# Create an empty list for your lidar sensor readings here,
# as well as an array that contains the angles of each ray 
# in radians. The total field of view is LIDAR_ANGLE_RANGE,
# and there are LIDAR_ANGLE_BINS. An easy way to generate the
# array that contains all the angles is to use linspace from
# the numpy package.
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS)
map_scale = 300  # Pixels per meter

#### End of Part 1 #####

# ------------------ MAIN CONTROL LOOP ------------------ #
while robot.step(SIM_TIMESTEP) != -1:
    
    #####################################################
    #                 Sensing                           #
    #####################################################
    
    # Read ground sensors
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()
    
    # Read Lidar           
    lidar_sensor_readings = lidar.getRangeImage() # rhos
    
    ##### Part 2: Turn world coordinates into map coordinates #####
    #
    # Come up with a way to turn the robot pose (in world coordinates)
    # into coordinates on the map. Draw a red dot using display.drawPixel()
    # where the robot moves.
    #
    # The arena is 1m x 1m and the display is 300x300 pixels.
    # Compute the robot’s pixel coordinates but delay drawing the red dot until after Part 4.
    robot_pixel_x = int(pose_x * map_scale)
    robot_pixel_y = int(pose_y * map_scale)
    
    ##### Part 3: Convert Lidar data into world coordinates #####
    #
    # Each Lidar reading has a distance rho and an angle alpha.
    # First compute the corresponding rx and ry of where the lidar
    # hits the object in the robot coordinate system. Then convert
    # rx and ry into world coordinates wx and wy. 
    # The arena is 1x1m2 and its origin is in the top left of the arena. 
    #
    # Use the homogeneous transformation:
    #    [ cos(theta) -sin(theta)  pose_x ]
    #    [ sin(theta)  cos(theta)  pose_y ]
    #    [     0           0          1   ]
    #
    # Note: With the new conversion, a beam with alpha = 0 now yields (0, rho)
    # in the robot frame, which—after transformation—adds to pose_y (forward).
    obstacle_points = []
    for i, rho in enumerate(lidar_sensor_readings):
        if math.isinf(rho) or rho > LIDAR_SENSOR_MAX_RANGE:
            obstacle_points.append(None)
        else:
            alpha = lidar_offsets[i]
            rx, ry = rho * math.sin(-alpha), rho * math.cos(-alpha)
            wx = pose_x + (rx * math.cos(pose_theta) - ry * math.sin(pose_theta))
            wy = pose_y + (rx * math.sin(pose_theta) + ry * math.cos(pose_theta))
            obstacle_points.append((wx, wy))
    
    ##### Part 4: Draw the obstacle and free space pixels on the map #####
    for point in obstacle_points:
        if point:
            wx, wy = point
            obstacle_pixel_x = int(wx * map_scale)
            obstacle_pixel_y = int(wy * map_scale)
            
            display.setColor(0xFFFFFF)  # White for free space
            display.drawLine(robot_pixel_x, robot_pixel_y, obstacle_pixel_x, obstacle_pixel_y)
            
            display.setColor(0x0000FF)  # Blue for obstacles
            display.drawPixel(obstacle_pixel_x, obstacle_pixel_y)
    
    display.setColor(0xFF0000)  # Red for robot pose
    display.drawPixel(robot_pixel_x, robot_pixel_y)
    # DO NOT CHANGE THE FOLLOWING CODE (You might only add code to display robot poses)
    #####################################################
    #                 Robot controller                  #
    #####################################################
    display.imageSave(None,"map.png") 
    if state == "line_follower":
        if(gsr[1]<350 and gsr[0]>400 and gsr[2] > 400):
            vL = MAX_SPEED*0.3
            vR = MAX_SPEED*0.3                
        # Checking for Start Line          
        elif(gsr[0]<500 and gsr[1]<500 and gsr[2]<500):
            vL = MAX_SPEED*0.3
            vR = MAX_SPEED*0.3
            # print("Over the line!") # Feel free to uncomment this
            display.imageSave(None,"map.png") 
        elif(gsr[2]<650): # turn right
            vL = 0.2*MAX_SPEED
            vR = -0.05*MAX_SPEED
        elif(gsr[0]<650): # turn left
            vL = -0.05*MAX_SPEED
            vR = 0.2*MAX_SPEED
             
    else:
        # Stationary State
        vL = 0
        vR = 0   
    
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
    
    #####################################################
    #                    Odometry                       #
    #####################################################
    
    EPUCK_MAX_WHEEL_SPEED = 0.11695*SIM_TIMESTEP/1000.0 
    dsr = vR/MAX_SPEED * EPUCK_MAX_WHEEL_SPEED
    dsl = vL/MAX_SPEED * EPUCK_MAX_WHEEL_SPEED
    ds = (dsr + dsl) / 2.0
    
    pose_y += ds * math.cos(pose_theta)
    pose_x += ds * math.sin(pose_theta)
    pose_theta += (dsr - dsl) / EPUCK_AXLE_DIAMETER
    
    # Feel free to uncomment this for debugging
    #print("X: %f Y: %f Theta: %f " % (pose_x, pose_y, pose_theta))