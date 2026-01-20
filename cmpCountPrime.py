import taichi as ti
import taichi.math as tm
import time

@ti.func
def ti_is_prime(num: ti.i32) -> ti.i32:
    is_p = 1
    if num < 2:
        is_p = 0
        pass
    else:
        for i in range(2, int(tm.sqrt(num)) + 1):
            if num % i == 0:
                is_p = 0
                break
    return is_p

@ti.kernel
def count_primes(n: ti.i32) -> ti.i32:
    count = 0
    for i in range(2, n):
        count += ti_is_prime(i)
    return count

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    # normal count prime
    n = 1000000
    start = time.time()
    count = 0
    for i in range(2, n):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    end = time.time()
    print(f"Normal count prime: {count} primes found in {end - start:.4f} seconds.")
    # taichi count prime
    n = 1000000
    start = time.time()
    count = count_primes(n)
    end = time.time()
    print(f"Taichi count prime: {count} primes found in {end - start:.4f} seconds.")