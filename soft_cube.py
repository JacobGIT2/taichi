import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.vulkan) 

n = 2
quad_size = 1.0 / n
dt = 4e-2 / 128 
substeps = int(1 / 20 / dt)

gravity = ti.Vector([0, -9.8, 0])
drag_damping = 0.5

spring_Y = 20000
dashpot_damping = 5
ground_stiffness = 50000

ground_height = -1.0

x = ti.Vector.field(3, dtype=float, shape=(n, n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n, n))

num_triangles = (n - 1) * (n - 1) * 2 * 6
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n * n)

# hard coded cube indices
data_list = [
    0, 1, 2, 1, 2, 3, 0, 1, 4, 1, 4, 5, 
    2, 3, 6, 3, 6, 7, 4, 5, 6, 5, 6, 7, 
    0, 2, 4, 2, 4, 6, 1, 3, 5, 3, 5, 7
]
arr = np.array(data_list, dtype=np.int32)
indices.from_numpy(arr)

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j, k in x:
        x[i, j, k] = [
            i * quad_size - 0.5 + random_offset[0], 
            j * quad_size + ground_height, 
            k * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j, k] = [0, 0, 0]

@ti.kernel
def initialize_mesh_color():
    for i in range(8):
        colors[i] = ti.Vector([0.22, 0.72, 0.52])

initialize_mesh_color()

spring_offsets = []
for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
    if i == 0 and j == 0 and k == 0:
        continue
    spring_offsets.append(ti.Vector([i, j, k]))

@ti.kernel
def substep(cur_time: float):
    breathing_scale = 1.2 + 0.4 * tm.sin(cur_time * 5.0)

    for i in ti.grouped(x):
        v[i] += gravity * dt


    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n and 0 <= j[2] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                current_dist = x_ij.norm()
                d = x_ij.normalized()
                
                original_dist = quad_size * float(spring_offset.norm())
                target_dist = original_dist * breathing_scale
                
                f_spring = -spring_Y * (current_dist / target_dist - 1)
                f_damping = -dashpot_damping * v_ij.dot(d)
                
                force += (f_spring + f_damping) * d

        v[i] += force * dt

    # ground collision
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        
        # soft ground collision
        if x[i][1] < ground_height:
            push_up = ground_height - x[i][1]
            v[i][1] += push_up * ground_stiffness * dt
            
            # horizontal friction
            v[i][0] = 0 
            v[i][2] = 0
            
            # ensure bounce back
            if v[i][1] < 0: v[i][1] *= -0.1 

        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j, k in ti.ndrange(n, n, n):
        vertices[i * n * n + j * n + k] = x[i, j, k]

window = ti.ui.Window("soft cube", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 100.0:
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep(current_t)
        current_t += dt
    
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()