### red -> bone
### blue -> soft
### cyan -> sin muscle
### orange -> cos muscle

import taichi as ti
import taichi.math as tm

from Ground import Ground
from _helper import generate_random_robot_numpy

ti.init(arch=ti.cuda)

# PHYSICS PARAMETERS
DT = 1e-3
SUBSTEPS = 10
GRAVITY = ti.Vector([0, -9.8, 0])
QUAD_SIZE = 0.04
BASE_STIFFNESS = 200000
DRAG_DAMPING = 1.5
DASHPOT_DAMPING = 0.6
ground_height = 0.0

@ti.data_oriented
class SoftRobot:
    def __init__(self, n: int):
        self.n = n
        self.n_verts = n + 1
        self.n_springs = 13 * n * n * n + 12 * n * n + 3 * n
        
        self.v_map = ti.field(dtype=int, shape=(n, n, n))
        
        # POSITION & VELOCITY
        num_physics_verts = self.n_verts * self.n_verts * self.n_verts
        
        self.x = ti.Vector.field(3, dtype=float, shape=num_physics_verts)
        self.v = ti.Vector.field(3, dtype=float, shape=num_physics_verts)
        
        # RENDER PARAMETERS
        self.render_x = ti.Vector.field(3, dtype=float, shape=n * n * n * 6 * 4)
        self.render_normal = ti.Vector.field(3, dtype=float, shape=n * n * n * 6 * 4)
        self.render_color = ti.Vector.field(3, dtype=float, shape=n * n * n * 6 * 4)
        self.render_indices = ti.field(dtype=int, shape=n * n * n * 6 * 6)
        
        self.num_render_verts = ti.field(int, shape=())
        self.num_render_indices = ti.field(int, shape=())

        # RENDER MAP: from render vertex idx to physics vertex idx, from render_x to x
        self.render_mapping = ti.field(int, shape=n * n * n * 6 * 4)

        # SPRING PARAMETERS
        self.num_springs = ti.field(int, shape=())
        self.spring_anchor_a = ti.field(int, shape=self.n_springs)
        self.spring_anchor_b = ti.field(int, shape=self.n_springs)
        self.spring_rest_len = ti.field(float, shape=self.n_springs)
        self.spring_stiffness = ti.field(float, shape=self.n_springs)
        self.spring_actuation = ti.field(float, shape=self.n_springs)
        self.spring_phase = ti.field(float, shape=self.n_springs) 
        
        self.material_stiffness = ti.field(dtype=float, shape=5)

    @ti.kernel
    def setup_materials(self):
        self.material_stiffness[0] = 3.0 * BASE_STIFFNESS 
        self.material_stiffness[1] = 0.5 * BASE_STIFFNESS 
        self.material_stiffness[2] = 1.0 * BASE_STIFFNESS 
        self.material_stiffness[3] = 1.0 * BASE_STIFFNESS 
        self.material_stiffness[4] = 0.0

    @ti.kernel
    def initialize_data(self):
        '''
        can be design as given input:
        color map: {mat_idx: color} int -> vec3
        '''
        ### initialize positions and velocities
        for i, j, k in ti.ndrange(self.n_verts, self.n_verts, self.n_verts):
            idx = i * (self.n_verts**2) + j * self.n_verts + k
            self.x[idx] = [
                i * QUAD_SIZE - self.n_verts * 0.5 * QUAD_SIZE,
                j * QUAD_SIZE + 0.5, 
                k * QUAD_SIZE - self.n_verts * 0.5 * QUAD_SIZE
            ]
            self.v[idx] = [0, 0, 0]

        ### initialize render parameters
        self.num_render_verts[None] = 0
        self.num_render_indices[None] = 0
        self.num_springs[None] = 0

        # initialize through each voxel: color (per voxel), normals (per face), indices (per face)
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            mat = self.v_map[i, j, k]
            if mat != 4:
                c = ti.Vector([0.8, 0.2, 0.2]) # red for material 0: bone
                if mat == 1: c = ti.Vector([0.2, 0.2, 0.8]) # blue for material 1: soft
                elif mat == 2: c = ti.Vector([0.2, 0.8, 0.8]) # cyan for material 2: sin muscle
                elif mat == 3: c = ti.Vector([0.9, 0.6, 0.1]) # orange for material 3: cos muscle
                
                # each face
                for d in range(6):
                    ni, nj, nk = i, j, k # neighbor idx
                    normal_vec = ti.Vector([0.0, 0.0, 0.0])
                    
                    if d == 0: ni -= 1; normal_vec = [-1, 0, 0]
                    elif d == 1: ni += 1; normal_vec = [1, 0, 0]
                    elif d == 2: nj -= 1; normal_vec = [0, -1, 0]
                    elif d == 3: nj += 1; normal_vec = [0, 1, 0]
                    elif d == 4: nk -= 1; normal_vec = [0, 0, -1]
                    elif d == 5: nk += 1; normal_vec = [0, 0, 1]

                    draw = False
                    if ni < 0 or ni >= self.n or nj < 0 or nj >= self.n or nk < 0 or nk >= self.n:
                        # neighbor out of bounds -> face exposed
                        draw = True
                    elif self.v_map[ni, nj, nk] == 4:
                        # neighbor is empty -> face exposed
                        draw = True
                    
                    if draw:
                        # add 4 vertices and 6 indices
                        base_v = ti.atomic_add(self.num_render_verts[None], 4)
                        base_i = ti.atomic_add(self.num_render_indices[None], 6)
                        
                        self.render_indices[base_i+0] = base_v + 0
                        self.render_indices[base_i+1] = base_v + 1
                        self.render_indices[base_i+2] = base_v + 2
                        self.render_indices[base_i+3] = base_v + 0
                        self.render_indices[base_i+4] = base_v + 2
                        self.render_indices[base_i+5] = base_v + 3

                        for v_local in range(4):
                            self.render_color[base_v + v_local] = c
                            self.render_normal[base_v + v_local] = normal_vec
                            
                            dx, dy, dz = 0, 0, 0
                            # hard coded local vertex offsets for each face
                            if d == 0: # -X
                                if v_local == 0: dx, dy, dz = 0, 0, 0
                                elif v_local == 1: dx, dy, dz = 0, 0, 1
                                elif v_local == 2: dx, dy, dz = 0, 1, 1
                                elif v_local == 3: dx, dy, dz = 0, 1, 0
                            elif d == 1: # +X
                                if v_local == 0: dx, dy, dz = 1, 0, 1
                                elif v_local == 1: dx, dy, dz = 1, 0, 0
                                elif v_local == 2: dx, dy, dz = 1, 1, 0
                                elif v_local == 3: dx, dy, dz = 1, 1, 1
                            elif d == 2: # -Y
                                if v_local == 0: dx, dy, dz = 0, 0, 1
                                elif v_local == 1: dx, dy, dz = 0, 0, 0
                                elif v_local == 2: dx, dy, dz = 1, 0, 0
                                elif v_local == 3: dx, dy, dz = 1, 0, 1
                            elif d == 3: # +Y
                                if v_local == 0: dx, dy, dz = 0, 1, 0
                                elif v_local == 1: dx, dy, dz = 0, 1, 1
                                elif v_local == 2: dx, dy, dz = 1, 1, 1
                                elif v_local == 3: dx, dy, dz = 1, 1, 0
                            elif d == 4: # -Z
                                if v_local == 0: dx, dy, dz = 1, 0, 0
                                elif v_local == 1: dx, dy, dz = 0, 0, 0
                                elif v_local == 2: dx, dy, dz = 0, 1, 0
                                elif v_local == 3: dx, dy, dz = 1, 1, 0
                            elif d == 5: # +Z
                                if v_local == 0: dx, dy, dz = 0, 0, 1
                                elif v_local == 1: dx, dy, dz = 1, 0, 1
                                elif v_local == 2: dx, dy, dz = 1, 1, 1
                                elif v_local == 3: dx, dy, dz = 0, 1, 1
                            
                            phys_idx = (i+dx)*(self.n_verts**2) + (j+dy)*self.n_verts + (k+dz)
                            self.render_mapping[base_v + v_local] = phys_idx

    @ti.kernel
    def update_render_data(self):
        count = self.num_render_verts[None]
        for i in range(count):
            phys_idx = self.render_mapping[i]
            self.render_x[i] = self.x[phys_idx]

    @ti.kernel
    def build_springs(self):
        self.num_springs[None] = 0
        
        # start vertex
        for i, j, k in ti.ndrange(self.n_verts, self.n_verts, self.n_verts):
            idx1 = i * (self.n_verts**2) + j * self.n_verts + k
            
            # only consider half of the 26 neighbors to avoid double counting
            neighbors = ti.Matrix([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],          # edge
                [1, 1, 0], [1, -1, 0], [1, 0, 1],         # face
                [1, 0, -1], [0, 1, 1], [0, 1, -1],        # face
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1] # body
            ])

            for d in range(13):
                dx, dy, dz = neighbors[d, 0], neighbors[d, 1], neighbors[d, 2]
                
                # neighbor vertex
                ni, nj, nk = i + dx, j + dy, k + dz

                if ni >= 0 and ni < self.n_verts and \
                   nj >= 0 and nj < self.n_verts and \
                   nk >= 0 and nk < self.n_verts:
                    
                    idx2 = ni * (self.n_verts**2) + nj * self.n_verts + nk
                    
                    # consider all voxels that share this spring
                    total_k = 0.0
                    # cartesian coordinates for calculating actuation
                    sum_x = 0.0
                    sum_y = 0.0

                    # check all 8 voxels around vertex (i, j, k)
                    for off_i, off_j, off_k in ti.static(ti.ndrange(2, 2, 2)):
                        # voxel index
                        vi, vj, vk = i - off_i, j - off_j, k - off_k

                        if vi >= 0 and vi < self.n and \
                           vj >= 0 and vj < self.n and \
                           vk >= 0 and vk < self.n:
                            
                            # (ni, nj, nk) is one of the 8 corners of voxel (vi, vj, vk)
                            if (ni == vi or ni == vi + 1) and \
                               (nj == vj or nj == vj + 1) and \
                               (nk == vk or nk == vk + 1):

                                mat = self.v_map[vi, vj, vk]
                                if mat != 4: # not empty
                                    k_val = self.material_stiffness[mat]
                                    
                                    # stiffness scaled based on edge
                                    contribution = k_val / 4.0
                                    
                                    total_k += contribution
                                    
                                    act, phase = 0.0, 0.0
                                    if mat == 2: 
                                        act, phase = 0.25, 0.0
                                    elif mat == 3: 
                                        act, phase = 0.25, 1.5707963
                                    
                                    if act > 0:
                                        # turn polar coordinates (act, phase) into cartesian
                                        sum_x += contribution * act * ti.cos(phase)
                                        sum_y += contribution * act * ti.sin(phase)

                    # spring exists
                    if total_k > 0:
                        sid = ti.atomic_add(self.num_springs[None], 1)
                        if sid < self.n_springs:
                            
                            dist = (self.x[idx1] - self.x[idx2]).norm()
                            
                            self.spring_anchor_a[sid] = idx1
                            self.spring_anchor_b[sid] = idx2
                            self.spring_rest_len[sid] = dist
                            self.spring_stiffness[sid] = total_k
                            final_x = sum_x / total_k
                            final_y = sum_y / total_k
                            
                            # final activation：sqrt(x^2 + y^2)
                            self.spring_actuation[sid] = ti.sqrt(final_x**2 + final_y**2)
                            
                            # use arctan2 to get correct phase angle
                            self.spring_phase[sid] = ti.atan2(final_y, final_x)
                        else:
                            ti.atomic_add(self.num_springs[None], -1)

    @ti.kernel
    def compute_forces(self, t: float):
        for i in range(self.n_verts * self.n_verts * self.n_verts):
            self.v[i] += GRAVITY * DT
            
        limit = self.num_springs[None]
        for i in range(limit):
            idx_a, idx_b = self.spring_anchor_a[i], self.spring_anchor_b[i]
            pos_a, pos_b = self.x[idx_a], self.x[idx_b]
            vel_a, vel_b = self.v[idx_a], self.v[idx_b]
            diff = pos_a - pos_b
            dist = diff.norm()
            if dist > 1e-4 and not tm.isnan(dist):
                dirc = diff / dist
                rest = self.spring_rest_len[i]
                act = self.spring_actuation[i]
                phase = self.spring_phase[i]
                target = rest * (1.0 + act * tm.sin(t * 15.0 + phase))
                stiff = self.spring_stiffness[i]
                v_rel = vel_a - vel_b
                force = (stiff * (dist - target) + DASHPOT_DAMPING * v_rel.dot(dirc)) * dirc
                self.v[idx_b] += force * DT
                self.v[idx_a] -= force * DT

    @ti.kernel
    def update_positions(self):
        for i in range(self.n_verts * self.n_verts * self.n_verts):
            self.v[i] *= tm.exp(-DRAG_DAMPING * DT)
            # ground collision
            if self.x[i].y < ground_height:
                self.x[i].y = ground_height
                if self.v[i].y < 0: self.v[i].y = 0
                self.v[i].x *= 0.8 
                if self.v[i].z > 0: self.v[i].z *= 0.98
                else: self.v[i].z *= 0.20
            
            self.x[i] += self.v[i] * DT
    
    def set_map(self, numpy_map):
        self.v_map.from_numpy(numpy_map)



#### MAIN SIMULATION LOOP
def main():
    N = 25
    robot = SoftRobot(N) # predefine robot size (N * N * N voxel space)
    robot.setup_materials()
    ground = Ground()
    ground.initialize()
    
    window = ti.ui.Window("Evolutionary Bot (Hard Surface)", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(2.0, 2.0, 2.0)
    camera.lookat(0.0, 0.2, 0.0)
    
    def reset_simulation():
        print(">>> Resetting simulation...")
        new_map = generate_random_robot_numpy(N)
        robot.set_map(new_map) 
        robot.initialize_data()
        robot.build_springs()
        return 0.0 

    sim_time = reset_simulation()

    while window.running:
        # if sim_time > 7.0:
        #     sim_time = reset_simulation()

        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                sim_time = reset_simulation()
        
        # update
        for _ in range(SUBSTEPS):
            robot.compute_forces(sim_time)
            robot.update_positions()
            sim_time += DT
        
        robot.update_render_data()

        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.point_light(pos=(0, 5, 5), color=(0.8, 0.8, 0.8))
        scene.point_light(pos=(0, 5, -5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        scene.mesh(ground.x, indices=ground.indices, normals=ground.normals, per_vertex_color=ground.colors)
        
        scene.mesh(
            robot.render_x, 
            indices=robot.render_indices, 
            normals=robot.render_normal, 
            per_vertex_color=robot.render_color, 
            index_count=robot.num_render_indices[None]
        )
        
        canvas.set_background_color((0.8, 0.9, 1.0))
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()