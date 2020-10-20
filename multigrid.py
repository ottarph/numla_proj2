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
    #inner = lambda x, y:np.inner(x.flatten(), y.flatten())

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

        ak = Nk / inner(A(pk), pk)
        #ak = inner(rk, rk) / inner(A(pk), pk)

        xkp = xk + ak * pk
        rkp = rk - ak * A(pk)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)

        bk = Nkp / Nk
        #bk = inner(rkp, rkp) / inner(rk, rk)
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

    fig = plt.figure() #figsize=(12, 8)
    ax = fig.add_subplot(111, projection='3d')

    K = np.linalg.cond(L)
    print(K)

    surf = ax.plot_surface(x, y, ( (L@u.flatten()).reshape(N,N) - b.reshape(N,N) ) * K,
                            rstride=1, cstride=1, # Sampling rates for the x and y input data
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


def my_cg(u0, f, N, tol=1e-12, max_iter=500):

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

    inner = lambda x, y: np.sum( (x * y) )

    r0 = f - L(u0)
    p0 = np.copy(r0)

    N0 = inner(r0, r0)
    n0 = np.sqrt(N0)

    uk = np.copy(u0)
    rk = r0
    pk = p0
    Nk = N0
    nk = n0

    i = 0
    while nk / n0 > tol and i < max_iter + 1:
        i += 1

        ak = Nk / inner(L(pk), pk)

        ukp = uk + ak * pk
        rkp = rk - ak * L(pk)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)

        bk = Nkp / Nk
        pkp = rkp + bk * pk

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

    #b = np.zeros((N+2, N+2), dtype=float)
    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    #b += 2
    
    u0 = np.zeros_like(b)
    u0 = g(x, y)
    #u0 = np.zeros((N+2, 2*N+1))
    u0[1:-1, 1:-1] = 0.5

    s = 2
    #print(u0)
    #u0[1:-1,1:-1] = 1
    u = np.copy(u0)
    u[::s, ::s], i = my_cg(u0[::s, ::s], b[::s, ::s], N, max_iter=1000)
    
    print(i)

    fig = plt.figure() #figsize=(12, 8)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x[::s,::s], y[::s,::s], u[::s,::s],
                            rstride=1, cstride=1, # Sampling rates for the x and y input data
                            cmap=matplotlib.cm.viridis)
    plt.show()

    return

def restriction(x, N):
    """ restriction with full weighting """
    y = np.zeros(int(N/2) + 1, dtype=float)



    return x[::2,::2]




    return



def mgv(u0, rhs, N, nu1, nu2, level, max_level):
    # the function mgv(u0,rhs,N,nu1,nu2,level,max_level) performs
    # one multigrid V-cycle on the 2D Poisson problem on the unit
    # square [0,1]x[0,1] with initial guess u0 and righthand side rhs.
    #
    # input: u0 - initial guess
    # rhs - righthand side
    # N - u0 is a (N+1)x(N+1) matrix
    # nu1 - number of presmoothings
    # nu2 - number of postsmoothings
    # level - current level
    # max_level - total number of levels
    #
    if level==max_level:
        u, resvec, i = my_cg(u0,rhs,N,1.e-13,500)
    else:
        u = jacobi(u0, rhs, 2/3, N, nu1)
        rf = residual(u, rhs, N)
        rc = restriction(rf, N)
        ec = mgv(np.zeros((int(N/2)+1,int(N/2)+1)), rc, int(N/2), nu1, nu2, level+1, max_level)
        ef = interpolation(ec, int(N/2))
        u = u + ef
        u = jacobi(u,rhs,2/3,N,nu2)
    return u



def main():

    test_my_cg()

    #test_conjugate_gradient()


    return


if __name__ == '__main__':
    main()
