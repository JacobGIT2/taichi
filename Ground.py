import taichi as ti

# FIXED GROUND PARAMETERS
GROUND_SIZE = 24
TILE_SIZE = 0.2
GROUND_VERTS = GROUND_SIZE * GROUND_SIZE * 4
GROUND_INDS = GROUND_SIZE * GROUND_SIZE * 6
ground_height = 0.0


### GROUND
@ti.data_oriented
class Ground:
    def __init__(self):
        self.x = ti.Vector.field(3, dtype=float, shape=GROUND_VERTS)
        self.colors = ti.Vector.field(3, dtype=float, shape=GROUND_VERTS)
        self.indices = ti.field(int, shape=GROUND_INDS)
        self.normals = ti.Vector.field(3, dtype=float, shape=GROUND_VERTS)

    @ti.kernel
    def initialize(self):
        offset = GROUND_SIZE * TILE_SIZE * 0.5
        for i, j in ti.ndrange(GROUND_SIZE, GROUND_SIZE):
            base_idx = (i * GROUND_SIZE + j) * 4
            ox, oz = i * TILE_SIZE - offset, j * TILE_SIZE - offset
            self.x[base_idx+0] = [ox, ground_height, oz]
            self.x[base_idx+1] = [ox+TILE_SIZE, ground_height, oz]
            self.x[base_idx+2] = [ox+TILE_SIZE, ground_height, oz+TILE_SIZE]
            self.x[base_idx+3] = [ox, ground_height, oz+TILE_SIZE]
            up = ti.Vector([0.0, 1.0, 0.0])
            for k in range(4): self.normals[base_idx+k] = up
            c = ti.Vector([0.9, 0.9, 0.9]) if (i+j)%2==0 else ti.Vector([0.5, 0.5, 0.5])
            for k in range(4): self.colors[base_idx+k] = c
            base_ind = (i*GROUND_SIZE+j)*6
            self.indices[base_ind+0] = base_idx+0
            self.indices[base_ind+1] = base_idx+1
            self.indices[base_ind+2] = base_idx+2
            self.indices[base_ind+3] = base_idx+0
            self.indices[base_ind+4] = base_idx+2
            self.indices[base_ind+5] = base_idx+3