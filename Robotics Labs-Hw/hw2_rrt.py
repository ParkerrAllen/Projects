import numpy as np
import matplotlib.pyplot as plt
import math
import random

###############################################################################
## Base Code
###############################################################################
class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.cost = float('inf')
def get_nd_obstacle(state_bounds):
    center_vector = []
    for d in range(state_bounds.shape[0]):
        center_vector.append(state_bounds[d][0] + random.random()*(state_bounds[d][1]-state_bounds[d][0]))
    radius = random.random() * 0.6
    return [np.array(center_vector), radius]

def setup_random_2d_world():
    state_bounds = np.array([[0,10],[0,10]])

    obstacles = [] # [pt, radius] circular obstacles
    for n in range(30):
        obstacles.append(get_nd_obstacle(state_bounds))

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid

def setup_fixed_test_2d_world():
    state_bounds = np.array([[0,1],[0,1]])
    obstacles = [] # [pt, radius] circular obstacles
    obstacles.append([np.array([0.5,0.5]),0.2])
    obstacles.append([np.array([0.1,0.7]),0.1])
    obstacles.append([np.array([0.7,0.2]),0.1])

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid


def plot_circle(x, y, radius, color="-k"):
    deg = np.linspace(0,360,50)

    xl = [x + radius * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + radius * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def get_random_valid_vertex(state_valid, bounds, obstacles):
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex

def visualize_2D_graph(state_bounds, obstacles, nodes, goal_point=None, filename=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    for obs in obstacles:
        plot_circle(obs[0][0], obs[0][1], obs[1])

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None: 
            if cur_node.parent is not None:
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

# END BASE CODE
#####################################


def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: n-dimensional array representing a point
    @return Node in node_list with closest node.point to query q_point
    '''
    # TODO: Your Code Here
    min_d=float('inf')
    nn=None
    for node in node_list:
        dis=np.linalg.norm(node.point-q_point)
        if dis<min_d: 
            min_d=dis
            nn=node
    return nn
def steer(from_point, to_point, delta_q):
    '''
    @param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    @param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''

    path = []

    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away
    # TODO Use the np.linspace function to get 10 points along the path from "from_point" to "to_point"
    d=to_point-from_point
    dis = np.linalg.norm(d)
    if dis>delta_q:
        d/=dis
        to_point=from_point+d*delta_q
    path = np.linspace(from_point, to_point, num=10)
    return path.tolist()


def rrt(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    TODO: Implement the RRT algorithm here, making use of the provided state_is_valid function. 
    If goal_point is set, your implementation should return once a path to the goal has been found 
    (e.g., if q_new.point is within 1e-5 distance of goal_point), using k as an upper-bound for iterations. 
    If goal_point is None, it should build a graph without a goal and terminate after k iterations.

    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param starting_point: Point within state_bounds to grow the RRT from
    @param goal_point: Point within state_bounds to target with the RRT. (OPTIONAL, can be None)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT graph nodes
    '''
    # I used AI if there was a error to see what was wrong, but most of the time I figured it out
    node_list = []
    node_list.append(Node(starting_point, parent=None)) # Add Node at starting point with no parent
    root=Node(starting_point, parent=None)
    node_list.append(root)
    for _ in range (k):
        if goal_point is not None and random.random()<0.05:
            goal=goal_point
        else:
            goal = np.random.rand(state_bounds.shape[0])*(state_bounds[:,1]-state_bounds[:,0])+state_bounds[:,0]
        nn=get_nearest_vertex(node_list, goal)
        path=steer(nn.point, goal, delta_q)
        if not all(state_is_valid(pt) for pt in path):
            continue
        nn=Node(np.array(path[-1]), parent=nn)
        nn.path_from_parent = path
        node_list.append(nn)
        if goal_point is not None and np.linalg.norm(nn.point -goal_point) <1e-5:
            break
    return node_list
    # TODO: Your code here
    # TODO: Make sure to add every node you create onto node_list, and to set node.parent and node.path_from_parent for each


def rrt_star(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    TODO: Implement the RRT* algorithm here, making use of the provided state_is_valid function

    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes
    '''
    # I used AI if there was a error to see what was wrong, but most of the time I figured it out
    node_list = []
    root=Node(starting_point)
    root.cost=0
    node_list.append(root)
    for _ in range (k):
        if goal_point is not None and random.random()<0.05:
            goal=goal_point
        else:
            goal=np.random.rand(state_bounds.shape[0])*(state_bounds[:,1]-state_bounds[:,0])+state_bounds[:,0]
        nn=get_nearest_vertex(node_list, goal)
        path=steer(nn.point, goal, delta_q)
        if not all(state_is_valid(point) for point in path):
            continue
        newn=Node(np.array(path[-1]), parent=nn)
        newn.path_from_parent=path
        newn.parent=nn
        newn.cost=nn.cost+np.linalg.norm(nn.point-newn.point)
        node_list.append(newn)
        r=1.5*delta_q
        nei=[n for n in node_list if np.linalg.norm(n.point-newn.point)<r]
        bp=nn
        bc=newn.cost
        bpath=path
        for n in nei:
            pc=n.cost+np.linalg.norm(n.point-newn.point)
            if pc<bc:
                test=steer(n.point, newn.point, delta_q)
                if all(state_is_valid(point) for point in test):
                    bp=n
                    bc=pc
                    bpath=test
        newn.parent=bp
        newn.path_from_parent=bpath
        newn.cost=bc
        node_list.append(newn)
        for n in nei:
            if n is not nn.parent:
                nc = nn.cost+np.linalg.norm(n.point-nn.point)
                if nc < n.cost:
                    test=steer(n.point, nn.point, delta_q)
                    if all(state_is_valid(point) for point in test):
                        n.parent=nn
                        n.path_from_parent=test
                        n.cost=nc
                        
        if goal_point is not None and np.linalg.norm(nn.point-goal_point) < 1e-5:
            break                                    
    # TODO: Your code here


    return node_list


if __name__ == "__main__":
    K = 500 # Feel free to adjust as desired
    ## Part 1: RRT
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = None
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, 'rrt_run1.png')

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, 'rrt_run2.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, 'rrt_goal_run1.png')

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, 'rrt_goal_run2.png')

    ## Part 2: RRT*
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, 'rrt_star_run1.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, 'rrt_star_run2.png')