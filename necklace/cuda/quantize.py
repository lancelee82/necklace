"""CUDA functions for matrix quantize (2bit) compression"""

import numpy as np

from numba import cuda


CMP_RATIO = 16

g = 0b01  # a >= thrd
e = 0b00  # -thrd < a < thrd
l = 0b10  # a <= -thrd
msk = 0b11  # or mask


@cuda.jit('(float32[:,:], uint32[:,:], float32[:,:], float32)')
def cu_knl_matrix_quantize_e(A, C, R, thrd=0.3):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    m = C.shape[0]
    n = A.shape[0]

    if x >= m or y >= n:
        return

    C[x, y] = 0
    for i in range(CMP_RATIO):
        xm = x * CMP_RATIO + i

        if xm >= n:
            break

        if A[xm, y] >= thrd:
            c = g
            R[xm, y] = A[xm, y] - thrd
        elif A[xm, y] <= -thrd:
            c = l
            R[xm, y] = A[xm, y] + thrd
        else:
            c = e
            R[xm, y] = A[xm, y]

        C[x, y] = C[x, y] | (c << (i * 2))

        #print(x, y, i, A[xm, y], c, C[x, y], R[xm, y])


@cuda.jit('(float32[:,:], uint32[:,:], float32[:,:], float32)')
def cu_knl_matrix_dequantize_e(A, C, R, thrd=0.3):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    m = C.shape[0]
    n = A.shape[0]

    if x >= m or y >= n:
        return

    for i in range(CMP_RATIO):
        xm = x * CMP_RATIO + i

        if xm >= n:
            break

        c = (C[x, y] >> (i * 2)) & msk

        if c == g:
            A[xm, y] = thrd
        elif c == l:
            A[xm, y] = -thrd
        else:  # c == e
            A[xm, y] = 0.0

        #A[xm, y] += R[xm, y]  # NOTE: for test

        #print(x, y, i, A[xm, y], c, C[x, y], R[xm, y])


# -------------------------------------------------------------------

def cu_fn_matrix_quantize_e(A, C, R, thrd=0.3,
                            #bpg=4, tpb=32,  # TODO: how to set whose ???
                            bpg=4, tpb=128,
                            to_device=False, stream=None):

    if not stream:
        stream = cuda.stream()

    with stream.auto_synchronize():
        if to_device:
            dA = cuda.to_device(A, stream)
            dR = cuda.to_device(R, stream)
            dC = cuda.to_device(C, stream)
        else:
            dA, dR, dC = A, R, C

        cu_knl_matrix_quantize_e[(bpg, bpg), (tpb / CMP_RATIO, tpb), stream](dA, dC, dR, thrd)

        #hR = dR.copy_to_host()
        #hC = dC.copy_to_host()


def cu_fn_matrix_dequantize_e(A, C, R, thrd=0.3,
                              #bpg=4, tpb=32,
                              bpg=4, tpb=128,
                              to_device=False, stream=None):

    if not stream:
        stream = cuda.stream()

    with stream.auto_synchronize():
        if to_device:
            dA = cuda.to_device(A, stream)
            dR = cuda.to_device(R, stream)
            dC = cuda.to_device(C, stream)
        else:
            dA, dR, dC = A, R, C

        cu_knl_matrix_dequantize_e[(bpg, bpg), (tpb / CMP_RATIO, tpb), stream](dA, dC, dR, thrd)


# -------------------------------------------------------------------

def test_11():

    from timeit import default_timer as time

    n = 7 * 32

    m = n / CMP_RATIO
    if n % CMP_RATIO != 0:
        m += 1

    thrd = 0.5

    A = np.array(np.random.random((n, n)), dtype=np.float32) - thrd * 2
    R = np.empty_like(A)  # for residual
    C = np.array(np.zeros((m, n)), dtype=np.uint32)

    stream = cuda.stream()
    dA = cuda.to_device(A, stream)
    dR = cuda.to_device(R, stream)
    dC = cuda.to_device(C, stream)

    tm = time()
    cu_fn_matrix_quantize_e(dA, dC, dR, thrd)
    print('quantize: ', time() - tm)


    B = np.empty_like(A)  # for dequantize
    dB = cuda.to_device(B, stream)

    tm = time()
    cu_fn_matrix_dequantize_e(dB, dC, dR, thrd)
    print('dequantize: ', time() - tm)


if __name__ == '__main__':
    test_11()

