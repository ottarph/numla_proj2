import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from multigrid import *


def poisson_system_lintrans(u):
    
    u = np.copy(u)

    N = u.shape[0] - 2
    h = 1 / (N+1)

    index = np.arange(1, N+1)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index-1, index)
    ixp_y = np.ix_(index+1, index)
    ix_ym = np.ix_(index, index-1) 
    ix_yp = np.ix_(index, index+1)
    
    u[ixy] = -( u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp] - 4*u[ixy]) / h**2

    return u

def poisson_system_lintrans_exact_test():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    Es = []
    Ns = np.logspace(1, 3.5, num=20, dtype=int)
    for N in Ns:
        print(N)
        x = np.outer(np.linspace(0, 1, N+2), np.ones(N+2))
        y = np.outer(np.ones(N+2), np.linspace(0, 1, N+2))

        b = np.zeros((N+2,N+2), dtype=float)
        b = g(x, y)
        b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

        u = u_ex(x, y)
        Lu = poisson_system_lintrans(u)

        h = 1 / (N+1)

        Es.append(np.linalg.norm(Lu - b) / h)

    print()
    plt.figure()
    plt.loglog(Ns, Es, 'k:x', label=r'$||r||_2 \, / \, h$')
    plt.axhline(y=1104, color='black', linewidth=0.6, alpha=0.7, label=r'$y=1104$')
    plt.legend()
    print(Es)


def main():


    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N = 100
    x = np.outer(np.linspace(0, 1, N+2), np.ones(N+2))
    y = np.outer(np.ones(N+2), np.linspace(0, 1, N+2))

    b = np.zeros((N+2,N+2), dtype=float)
    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    print("\n----------\nb:")
    print(b.round(0))

    u = u_ex(x, y)
    Lu = poisson_system_lintrans(u)
    #print("\n----------\nu:")
    #print(u.round(2))
    print("\n----------\nLu:")
    print(Lu.round(0))

    x_cg, i = conjugate_gradient(poisson_system_lintrans, b, np.zeros((N+2,N+2)))

    fig = plt.figure() #figsize=(12, 8)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, u - x_cg,
                            rstride=1, cstride=1, # Sampling rates for the x and y input data
                            cmap=matplotlib.cm.viridis)
    plt.show()


    poisson_system_lintrans_exact_test()

    plt.show()

    return


if __name__ == '__main__':
    main()
