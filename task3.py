import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from multigrid import *
from time import time

def cg_pcg_comparison():
    
    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)

    cg_iters = []
    cg_times = []
    pcg_iters = []
    pcg_times = []

    Ns = [2**i for i in range(5, 7+1)]
    for N in Ns:
        print(N)

        x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
        y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

        rhs = g(x, y)
        rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

        np.random.seed(seed=1)
        u0 = np.copy(rhs)
        u0[1:-1,1:-1] = 1 - np.random.random((N-1, N-1))*0.5
        #u0[1:-1,1:-1] = 0.5


        """ Conjugate Gradient """
        tol      = 1e-13
        max_iter = 1000

        start = time()
        uh, i = my_cg(u0, rhs, N, tol=tol, max_iter=max_iter)
        end = time()
        cg_iters.append(i)
        cg_times.append(end - start)


        """ Preconditioned Conjugate Gradient """
        levels     = 3
        nu1        = 5
        nu2        = 5
        tol        = 1e-13
        max_iter   = 1000
        cg_tol     = 1e-3
        cg_maxiter = 500

        start = time()
        uh, i, ns = pcg(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
        end = time()
        pcg_iters.append(i)
        pcg_times.append(end - start)

    Ns = np.array(Ns, dtype=int)
    cg_iters = np.array(cg_iters, dtype=float)
    cg_times = np.array(cg_times, dtype=float)
    pcg_iters = np.array(cg_iters, dtype=float)
    pcg_times = np.array(cg_times, dtype=float)

    plt.figure()
    plt.semilogy(Ns, cg_iters, 'k--', label='CG')
    plt.semilogy(Ns, pcg_iters, 'k:', label='PCG')
    plt.ylabel('Iterations')
    plt.xlabel('$N$')
    plt.legend()

    plt.figure()
    plt.semilogy(Ns, cg_times, 'k--', label='CG')
    plt.semilogy(Ns, pcg_times, 'k:', label='PCG')
    plt.ylabel('Runtime')
    plt.xlabel('$N$')
    plt.legend()

    plt.show()

    return


def main():

    cg_pcg_comparison()

    return

if __name__ == '__main__':
    main()
