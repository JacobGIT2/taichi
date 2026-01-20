import random
import numpy as np

### RANDOM ROBOT GENERATOR
def generate_random_robot_numpy(N):
    map_data = np.full((N, N, N), 4, dtype=np.int32)
    start_y = 2
    current = [(N//2, start_y, N//2)] # start from middle
    material_dist = [2,3] 
    py_map = {(N//2, start_y, N//2): random.choice(material_dist)}
    dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    attempts = 0
    target_voxels = random.randint(2, 2)
    
    while len(py_map) < target_voxels and attempts < 10000:
        attempts += 1
        p = random.choice(current)
        if random.random() < 0.4: d = random.choice([(0,0,1), (0,0,-1)])
        else: d = random.choice(dirs)
        c = (p[0]+d[0], p[1]+d[1], p[2]+d[2]) # randomly grow from existing voxel
        # boundary check: filtered out
        if 1 <= c[0] < N-1 and 1 <= c[1] < N-1 and 1 <= c[2] < N-1:
            if c not in py_map:
                mat = random.choice(material_dist)
                py_map[c] = mat
                current.append(c)
    print(f"Generated robot with {len(py_map)} voxels.")
    for p, m in py_map.items():
        map_data[p[0], p[1], p[2]] = m
    return map_data

def random_reset_simulation(robot):
    print(">>> Randomly resetting simulation...")
    new_map = generate_random_robot_numpy(robot.n)
    robot.set_map(new_map) 
    robot.initialize_data()
    robot.build_springs()
    return 0.0

def reset_simulation(robot, new_map):
    print(">>> Resetting simulation...")
    robot.set_map(new_map) 
    robot.initialize_data()
    robot.build_springs()
    return 0.0