import numpy as np
from tests import *


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


def restriction(x, N):
    """ restriction with full weighting """
    n = int(N/2)
    y = np.zeros((n+1, n+1), dtype=float)

    index = np.arange(1, n)

    Ixy = np.ix_(index, index)

    ixy = np.ix_(2*index, 2*index)
    ixm_y = np.ix_(2*index-1, 2*index)
    ixp_y = np.ix_(2*index+1, 2*index)
    ix_ym = np.ix_(2*index, 2*index-1)
    ix_yp = np.ix_(2*index, 2*index+1)
    ixm_ym = np.ix_(2*index-1, 2*index-1)
    ixm_yp = np.ix_(2*index-1, 2*index+1)
    ixp_ym = np.ix_(2*index+1, 2*index-1)
    ixp_yp = np.ix_(2*index+1, 2*index+1)

    y[Ixy] = 1/16 * ( 4*x[ixy] + 2*(x[ixm_y] + x[ixp_y] + x[ix_ym] + x[ix_yp]) +
                        (x[ixm_ym] + x[ixm_yp] + x[ixp_ym] + x[ixp_yp]) )

    BDindex = np.array([0, n], dtype=int)
    BIxy = np.ix_(index, index)
    Bixy = np.ix_(2*index, 2*index)

    y[BIxy] = x[Bixy]

    return y


def interpolation(x, n):
    """ Linear interpolation to grid with half grid size"""
    N = 2*n
    y = np.zeros((2*n+1, 2*n+1), dtype=float)

    index = np.arange(0, n)

    Ixy = np.ix_(index, index)
    Ixp_y = np.ix_(index+1, index)
    Ix_yp = np.ix_(index, index+1)
    Ixp_yp = np.ix_(index+1, index+1)

    ixy = np.ix_(2*index, 2*index)
    ixp_y = np.ix_(2*index+1, 2*index)
    ix_yp = np.ix_(2*index, 2*index+1)
    ixp_yp = np.ix_(2*index+1, 2*index+1)

    y[ixy]    = x[Ixy]
    y[ixp_y]  = 1/2 * (x[Ixy] + x[Ixp_y])
    y[ix_yp]  = 1/2 * (x[Ixy] + x[Ix_yp])
    y[ixp_yp] = 1/4 * (x[Ixy] + x[Ixp_y] + x[Ix_yp] + x[Ixp_yp])

    """ Not sure if this section is needed """
    Edge = np.arange(0, 2*n + 1, step=2, dtype=int)
    edge = np.arange(0, n + 1, dtype=int)
    y[Edge, np.full(n+1, 0, dtype=int)] = x[edge, np.full(n+1, 0, dtype=int)]
    y[Edge, np.full(n+1, 2*n, dtype=int)] = x[edge, np.full(n+1, n, dtype=int)]
    y[np.full(n+1, 0, dtype=int), Edge] = x[np.full(n+1, 0, dtype=int), edge]
    y[np.full(n+1, 2*n, dtype=int), Edge] = x[np.full(n+1, n, dtype=int), edge]
    """ Should enforce accurate boundary conditions? """

    return y


def residual(u, rhs, N):

    def L(u):
        u = np.copy(u)

        h = 1 / N

        index = np.arange(1, N)
        ixy = np.ix_(index, index)
        ixm_y = np.ix_(index-1, index)
        ixp_y = np.ix_(index+1, index)
        ix_ym = np.ix_(index, index-1)
        ix_yp = np.ix_(index, index+1)
        
        u[ixy] = -( u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp] - 4*u[ixy]) / h**2

        return u

    return rhs - L(u)


def jacobi(u0, rhs, w, N, nu):

    h = 1 / N

    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index-1, index)
    ixp_y = np.ix_(index+1, index)
    ix_ym = np.ix_(index, index-1)
    ix_yp = np.ix_(index, index+1)

    def jacobi_step(uk, rhs, w, N):
        
        left_term = np.copy(uk)

        left_term[ixy] = 0.25 * ( uk[ixm_y] + uk[ixp_y] + uk[ix_ym] + uk[ix_yp] + h**2 * rhs[ixy] )

        ukp = w * left_term + (1-w) * uk

        return ukp

    uk = np.copy(u0)

    for _ in range(nu):
        uk = jacobi_step(uk, rhs, w, N)

    return uk


def mgv(u0, rhs, N, nu1, nu2, level, max_level, cg_tol=1e-13, cg_maxiter=500):
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
    print('yee')
    if level==max_level:
        #u, resvec, i = my_cg(u0,rhs,N,1.e-13,500)
        u, i = my_cg(u0, rhs, N, cg_tol, cg_maxiter)
        #print(i)
    else:
        u = jacobi(u0, rhs, 2/3, N, nu1)
        rf = residual(u, rhs, N)
        rc = restriction(rf, N)
        ec = mgv(np.zeros((int(N/2)+1,int(N/2)+1)), rc, int(N/2), nu1, nu2, level+1, max_level, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
        ef = interpolation(ec, int(N/2))
        u = u + ef
        u = jacobi(u, rhs, 2/3, N, nu2)
    return u


def mgv_debug(u0, rhs, N, nu1, nu2, level, max_level, uh_arr, cg_tol=1e-13, cg_maxiter=500):
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
        #u, resvec, i = my_cg(u0,rhs,N,1.e-13,500)
        u, i = my_cg(u0, rhs, N, cg_tol, cg_maxiter)
        #print(i)
    else:
        u = jacobi(u0, rhs, 2/3, N, nu1)
        rf = residual(u, rhs, N)
        rc = restriction(rf, N)
        ec = mgv_debug(np.zeros((int(N/2)+1,int(N/2)+1)), rc, int(N/2), nu1, nu2, level+1, max_level, uh_arr, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
        ef = interpolation(ec, int(N/2))
        u = u + ef
        u = jacobi(u, rhs, 2/3, N, nu2)
    uh_arr.append(u)
    return u


def pcg(u0, rhs, N, nu1, nu2, level, max_level, tol=1e-12, max_iter=500, cg_tol=1e-13, cg_maxiter=500):

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

    r0 = rhs - L(u0)
    z0 = mgv(np.zeros_like(r0), r0, N, nu1, nu2, level, max_level, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    p0 = np.copy(z0)

    N0 = inner(r0, r0)
    n0 = np.sqrt(N0)
    g0 = inner(r0, z0)

    uk = np.copy(u0)
    rk = r0
    zk = z0
    pk = p0
    Nk = N0
    nk = n0
    gk = g0

    ns = [nk]

    i = 0
    while nk / n0 > tol and i < max_iter + 1:
        i += 1

        ak = gk / inner(L(pk), pk)

        ukp = uk + ak * pk
        rkp = rk - ak * L(pk)
        zkp = mgv(np.zeros_like(rkp), rkp, N, nu1, nu2, level, max_level, cg_tol=cg_tol, cg_maxiter=cg_maxiter)

        Nkp = inner(rkp, rkp)
        nkp = np.sqrt(Nkp)
        gkp = inner(rkp, zkp)


        bk = gkp / gk
        pkp = zkp + bk * pk

        uk = ukp
        rk = rkp
        zk = zkp
        pk = pkp
        Nk = Nkp
        nk = nkp
        gk = gkp

        ns.append(nk)

    if i == max_iter + 1:
        raise Exception("Did not converge within maximum number of iterations")

    return uk, i, ns


def main():

    #test_my_cg()

    #test_restriction()

    #test_interpolation()

    #test_residual()

    test_jacobi()

    #test_mgv()


    return


if __name__ == '__main__':
    main()
