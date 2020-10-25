import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from multigrid import *


def test_conjugate_gradient():
    M = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)

    def A(x):

        return M @ x

    b = M @ np.array([1,2,3])

    print(b)
    print(A(b))

    x, i = conjugate_gradient(A, b, np.zeros(3))
    print()
    print(i)
    print(x)
    print(b - A(x))


    def A(x):
        return 0.5 * x
    b = A(np.array([[1, 2], [3, 4]], dtype=float))
    x, i = conjugate_gradient(A, b, np.zeros((2,2)))
    print()
    print(i)
    print(x)
    print(b - A(x))

    def A(x):
        return np.arange(1,5).reshape(2,2) * x
    b = A(np.array([[1, 2], [3, 4]], dtype=float))
    x, i = conjugate_gradient(A, b, np.zeros((2,2)))
    print()
    print(i)
    print(x)
    print(b - A(x))

    N = 20
    h = 1 / (N-1)
    L = np.zeros((N**2, N**2), dtype=float)
    B = np.diag(np.full(N, 4, dtype=float), 0)
    B += np.diag(np.full(N-1, -1, dtype=float),  1)
    B += np.diag(np.full(N-1, -1, dtype=float), -1)
    I = np.eye(N)

    L[0*N:(0+1)*N, 0*N:(0+1)*N] += B
    for i in range(1, N):
        L[i*N:(i+1)*N, i*N:(i+1)*N] += B
        L[(i-1)*N:i*N, i*N:(i+1)*N] -= I
        L[i*N:(i+1)*N, (i-1)*N:i*N] -= I

    L = L / h**2

    print(L)

    def A(x):

        return L @ x

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    x = np.outer(np.linspace(0, 1, N), np.ones(N))
    y = np.outer(np.ones(N), np.linspace(0, 1, N))

    b = np.zeros((N,N), dtype=float)
    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    b = b.flatten()

    u, i = conjugate_gradient(A, b, np.zeros_like(b))
    print(i)

    u = u.reshape(N,N)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    K = np.linalg.cond(L)
    print(K)

    surf = ax.plot_surface(x, y, ( (L@u.flatten()).reshape(N,N) - b.reshape(N,N) ) * K,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)
    plt.show()


    def L(u):
        v = np.zeros_like(u)

        v[0,0] = 2*u[0,0] - u[0,1]
        v[0,1] = -u[0,0] + 2*u[0,1] - u[1,0]
        v[1,0] = -u[0,1] + 2*u[1,0] - u[1,1]
        v[1,1] = -u[1,0] + 2*u[1,1]

        return v

    def L(u):
        A = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], dtype=float)

        #print(A.shape)
        #print(u.shape)

        uu = u.flatten()
        #print(uu.shape)
        #print(uu)

        return ( A @ uu ).reshape(u.shape)

    u0 = np.zeros((4,4), dtype=float)
    b_red = np.array([[10, 18], [17, 10]], dtype=float)
    b_red = np.array([[1, 0], [0, 1]], dtype=float)
    #u0 = np.zeros(4, dtype=float)
    #b_red = np.array([1, 0, 0, 1], dtype=float)
    u0[::2, ::2], i = conjugate_gradient(L, b_red, np.zeros_like(b_red))
    #u, i = conjugate_gradient(L, b_red, np.zeros_like(b_red))
    #print(i)
    #print(u)
    print(i)
    print(u0)



    return

def test_my_cg():

    def L(u):
        u = np.copy(u)

        N = u.shape[0] - 1
        h = 1 / N

        index = np.arange(1, N)
        ixy = np.ix_(index, index)
        ixm_y = np.ix_(index-1, index)
        ixp_y = np.ix_(index+1, index)
        ix_ym = np.ix_(index, index-1)
        ix_yp = np.ix_(index, index+1)
        
        u[ixy] = -( u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp] - 4*u[ixy]) / h**2

        return u

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y) + 1
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    #fp = lambda x, y: 2*x**2 - 2*x + 2*y**2 - 2*y
    #f = lambda x, y: 20*fp(x,y)
    #g = lambda x, y: 0*x*y
    #u_exp = lambda x, y: (x-x**2) * (y-y**2)
    #u_ex = lambda x, y: 20*u_exp(x,y)
    

    N = 2**8

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    
    u0 = np.zeros_like(b)
    u0 = g(x, y)
    u0[1:-1, 1:-1] = 0.5

    s = 2
    u = np.copy(u0)
    u[::s, ::s], i = my_cg(u0[::s, ::s], b[::s, ::s], N, max_iter=1000)
    
    print(i)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x[::s,::s], y[::s,::s], u[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)
    plt.show()

    return

def test_restriction():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N = 2**6

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    a = g(x, y)
    a[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    

    surf = ax.plot_surface(x, y, a,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    b = restriction(a, N)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = np.copy(x[::2,::2]), np.copy(y[::2,::2])

    surf = ax.plot_surface(xx, yy, b,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    c = restriction(b, int(N/2))

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = np.copy(x[::4,::4]), np.copy(y[::4,::4])

    surf = ax.plot_surface(xx, yy, c,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()

    return

def test_interpolation():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    n = 2**4

    x = np.outer(np.linspace(0, 1, n+1), np.ones(n+1))
    y = np.outer(np.ones(n+1), np.linspace(0, 1, n+1))

    a = g(x, y)
    a[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    a = u_ex(x,y)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, a,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    b = interpolation(a, n)

    x = np.outer(np.linspace(0, 1, 2*n+1), np.ones(2*n+1))
    y = np.outer(np.ones(2*n+1), np.linspace(0, 1, 2*n+1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, b,
                            rstride=1, cstride=1,
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, u_ex(x,y) - b,
                            rstride=1, cstride=1,
                            cmap=matplotlib.cm.viridis)

    plt.show()

    return

def test_residual():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N = 2**5

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    a = g(x, y)
    a[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    u = u_ex(x, y)

    r = residual(u, a, N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, r,
                            rstride=1, cstride=1,
                            cmap=matplotlib.cm.viridis)

    u0 = np.zeros_like(a)    
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    uh, i = my_cg(u0, a, N)

    rh = residual(uh, a, N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, rh,
                            rstride=1, cstride=1,
                            cmap=matplotlib.cm.viridis)

    plt.show()

    return

def test_jacobi():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N      = 2**5
    nu     = 5
    w      = 2/3

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1,N-1)) * 2 - 1

    uh = jacobi(u0, rhs, w, N, nu)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, u0,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, uh,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()


    return

def test_mgv():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N      = 2**10
    levels = 4
    nu1    = 20
    nu2    = 20

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    u = u_ex(x, y)
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))

    uh = mgv(u0, rhs, N, nu1, nu2, level=1, max_level=levels)


    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    s = 8
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], uh[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()


    return

def test_pcg():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)

    N          = 2**6
    levels     = 3
    nu1        = 10
    nu2        = 10
    tol        = 1e-13
    max_iter   = 40
    cg_tol     = 1e-8
    cg_maxiter = 300

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    u = u_ex(x, y)
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))

    uh, i = pcg(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)

    print(f'#iterations = {i}')

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    s = 2
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], uh[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()



    return

def main():

    #test_my_cg()

    #test_restriction()

    #test_interpolation()

    #test_residual()

    #test_jacobi()

    test_mgv()

    #test_pcg()

if __name__ == '__main__':
    main()