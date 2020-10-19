import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#from task1 import poisson_system_lintrans

def conjugate_gradient(A, b, x0, tol=1e-12, max_iter=500):
    """
    Presumed to work on 2D systems
        A:        function returning result of linear operation A(x)
        b:        RHS of linear system A(x) = b
        x0:       Starting guess
        tol:      Relative residual tolerance
        max_iter: Maximum number of iterations
    """

    inner = lambda x, y: np.sum( (x * y) )

    r0 = b - A(x0)
    p0 = np.copy(r0)

    #N0 = np.linalg.norm(r0, ord='fro')
    N0 = inner(r0, r0)
    n0 = np.sqrt(N0)

    xk = np.copy(x0)
    rk = r0
    pk = p0
    Nk = N0
    nk = n0

    i = 0
    while nk / n0 > tol and i < max_iter + 1:
        i += 1

        ak = Nk / inner(A(rk), rk)

        xkp = xk + ak * pk
        rkp = rk - ak * A(pk)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)

        bk = Nkp / Nk
        pkp = rkp + bk * pk

        xk = xkp
        rk = rkp
        pk = pkp
        Nk = Nkp
        nk = nkp

    if i == max_iter + 1:
        raise Exception("Did not converge within maximum number of iterations")

    return xk, i

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

    


    return


def my_cg(u0, f, N, tol=1e-12, max_iter=500):

    def L(u):
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

    inner = lambda x, y: np.sum( (x * y) )

    r0 = f - L(u0)
    p0 = np.copy(r0)

    #N0 = np.linalg.norm(r0, ord='fro')
    N0 = inner(r0, r0)
    n0 = np.sqrt(N0)

    uk = np.copy(u0)
    rk = r0
    pk = p0
    Nk = N0
    nk = n0

    i = 0
    while nk / n0 > tol and i < max_iter + 1:
        print(nk / n0)
        i += 1

        ak = Nk / inner(L(rk), rk)

        ukp = uk + ak * pk
        rkp = rk - ak * L(pk)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)

        fk = Nkp / Nk
        pkp = rkp + fk * pk

        uk = ukp
        rk = rkp
        pk = pkp
        Nk = Nkp
        nk = nkp

    if i == max_iter + 1:
        raise Exception("Did not converge within maximum number of iterations")

    return uk, i

def test_my_cg():

    def L(u):
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

    #f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    #g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    #u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    fp = lambda x, y: 2*x**2 - 2*x + 2*y**2 - 2*y
    f = lambda x, y: 20*fp(x,y)
    g = lambda x, y: 0*x*y
    u_exp = lambda x, y: (x-x**2) * (y-y**2)
    u_ex = lambda x, y: 20*u_exp(x,y)
    

    N = 100

    x = np.outer(np.linspace(0, 1, N+2), np.ones(N+2))
    y = np.outer(np.ones(N+2), np.linspace(0, 1, N+2))

    b = np.zeros((N+2,N+2), dtype=float)
    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    #b += 2
    

    u, i = my_cg(np.zeros_like(b), b, N)
    print(i)

    fig = plt.figure() #figsize=(12, 8)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, L(u_ex(x,y)),
                            rstride=1, cstride=1, # Sampling rates for the x and y input data
                            cmap=matplotlib.cm.viridis)
    plt.show()

    return


def other_cg(u0, f, N, tol=1e-12, max_iter=500):

    def L(u):
        v = np.zeros_like(u)

        v[0,0] = u[0,0] + 2*u[0,1] + 3*u[1,0] + 4*u[1,1]
        v[0,1] = 5*u[0,1] + 6*u[1,0] + 7*u[1,1]
        v[1,0] = 8*u[1,0] + 9*u[1,1]
        v[1,1] = 10*u[1,1]

        return v

    inner = lambda x, y: np.sum( (x * y) )

    def inner(x, y):
        print(x, y, sep='\n', end='\n\n')
        input()
        m, n = x.shape
        summ = 0

        for i in range(m):
            for j in range(n):
                summ += x[i,j] * y[i,j]

        return summ

    r0 = f - L(u0)
    p0 = np.copy(r0)

    #N0 = np.linalg.norm(r0, ord='fro')
    N0 = inner(r0, r0)
    n0 = np.sqrt(N0)

    uk = np.copy(u0)
    rk = r0
    pk = p0
    Nk = N0
    nk = n0

    i = 0
    while nk / n0 > tol and i < max_iter + 1:
        print(nk / n0)
        i += 1

        ak = Nk / inner(L(rk), rk)

        ukp = uk + ak * pk
        rkp = rk - ak * L(pk)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)

        fk = Nkp / Nk
        pkp = rkp + fk * pk

        uk = ukp
        rk = rkp
        pk = pkp
        Nk = Nkp
        nk = nkp

    if i == max_iter + 1:
        raise Exception("Did not converge within maximum number of iterations")

    return uk, i

def test_other_cg():

    x_ex = np.ones((2,2), dtype=float)

    x0 = np.zeros_like(x_ex)

    b = np.array([[10, 18], [17, 10]], dtype=float)

    def L(u):
        v = np.zeros_like(u)

        v[0,0] = u[0,0] + 2*u[0,1] + 3*u[1,0] + 4*u[1,1]
        v[0,1] = 5*u[0,1] + 6*u[1,0] + 7*u[1,1]
        v[1,0] = 8*u[1,0] + 9*u[1,1]
        v[1,1] = 10*u[1,1]

        return v

    print(L(x_ex))
    print(b)

    print('\n\n\n')

    x, i = other_cg(x0, b, 2)
    #x, i = conjugate_gradient(L, b, x0)
    print(i)

    print(x)


    return



def main():

    #test_my_cg()

    test_conjugate_gradient()

    test_other_cg()
    

    return


if __name__ == '__main__':
    main()
