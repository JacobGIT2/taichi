import taichi as ti
import taichi.math as tm

print(ti.gpu)
ti.init(arch=ti.gpu)