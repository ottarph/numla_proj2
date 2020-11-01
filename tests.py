import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from multigrid import *
from time import time


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
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    #fp = lambda x, y: 2*x**2 - 2*x + 2*y**2 - 2*y
    #f = lambda x, y: 20*fp(x,y)
    #g = lambda x, y: 0*x*y
    #u_exp = lambda x, y: (x-x**2) * (y-y**2)
    #u_ex = lambda x, y: 20*u_exp(x,y)
    

    N = 2**7

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    b = g(x, y)
    b[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    
    u0 = np.zeros_like(b)
    u0 = g(x, y)
    np.random.seed(seed=1)
    u0[1:-1, 1:-1] = 1 - 0.5*np.random.random((N-1,N-1))

    s = 1
    u = np.copy(u0)
    u[::s, ::s], i, ns = my_cg(u0[::s, ::s], b[::s, ::s], N, max_iter=1000)
    
    print(i)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    s = 1
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], u[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    s = 1
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], (u_ex(x,y) - u)[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    '''
    errs = []
    for _ in range(10):
        print(_)
        u0 = np.zeros_like(b)
        u0 = g(x, y)
        u0[1:-1, 1:-1] = 1 - 0.5*np.random.random((N-1,N-1))

        u, i = my_cg(u0, b, N, max_iter=1000)

        kl = np.unravel_index(np.argmax((u_ex(x,y) - u)), u.shape)
        #print(kl)
        print(x[kl], y[kl])
        print(u_ex(x, y)[kl] - u[kl])

        e = np.linalg.norm((u_ex(x,y) - u).flatten(), ord=np.inf)
        errs.append(e)

    fig, ax = plt.subplots()
    plt.plot(range(len(errs)), errs, 'k:')
    print(errs)
    '''

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
    print(a.shape)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    

    surf = ax.plot_surface(x, y, a,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    b = restriction(a, N)
    print(b.shape)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = np.copy(x[::2,::2]), np.copy(y[::2,::2])

    surf = ax.plot_surface(xx, yy, b,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    c = restriction(b, int(N/2))
    print(c.shape)

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
    uh, i, ns = my_cg(u0, a, N)

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
    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)

    N      = 2**5
    nu     = 50
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
    ax.set_title(r"$\mathrm{rhs}$")
    surf = ax.plot_surface(x, y, rhs,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("$u_0$")
    surf = ax.plot_surface(x, y, u0,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("$u_h$")
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
    ax.set_title(r"$u_h$")
    s = 8
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], uh[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r"$u - u_h$")
    s = 8
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], (u - uh)[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()


    return

def test_mgv_poly():

    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)
    

    N      = 2**7
    levels = 4
    nu1    = 40
    nu2    = 40

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    
    uh = mgv(u0, rhs, N, nu1, nu2, level=1, max_level=levels)

    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    s = 2
    surf = ax.plot_surface(x[::s,::s], y[::s,::s], uh[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis)

    plt.show()


    return

def test_mgv_iteration():

    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)
    

    N          = 3*2**7
    levels     = 4
    nu1        = 5
    nu2        = 5
    tol        = 1e-12
    max_iter   = 20
    cg_tol     = 1e-3
    cg_maxiter = 400

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    
    start = time()
    uh, i, ns = mgv_iteration(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    end = time()
    runtime = end - start

    print(f'#MG iterations = {i}')
    print(f'MG runtime = {runtime:.2f}')

    start = time()
    uh_cg, i_cg, ns_cg = my_cg(u0, rhs, N, tol=tol, max_iter=2000)
    end = time()
    runtime_cg = end - start
    print(f'#CG iterations = {i_cg}')
    print(f'CG runtime = {runtime_cg:.2f}')

    plt.figure()
    ns = np.array(ns, dtype=float)
    ns_cg = np.array(ns_cg, dtype=float)
    plt.semilogy(range(len(ns)), ns / ns[0], 'k:', label=r'MG V-cycle')
    plt.semilogy(range(len(ns_cg)), ns_cg / ns_cg[0], 'k--', label=r'Conjugate gradient')
    plt.legend()

    plt.show()

    return

def test_mgv_iteration_steps():

    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)
    

    N          = 2**5
    levels     = 4
    nu1        = 2
    nu2        = 2
    tol        = 1e-12
    max_iter   = 100
    cg_tol     = 1e-7
    cg_maxiter = 400

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    
    start = time()
    uh, i, ns, uh_arr = mgv_iteration_steps(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    end = time()
    runtime = end - start

    K = len(uh_arr[:5+1])
    s = 2
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i, uh in enumerate(uh_arr[:5+1]):
        k = uh.shape[0]
        xx = np.outer(np.linspace(0, 1, k), np.ones(k))
        yy = np.outer(np.ones(k), np.linspace(0, 1, k))
        ax = fig.add_subplot(K//2 + K%2, 2, i+1, projection='3d')
        surf = ax.plot_surface(xx, yy, uh,
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis, label=f'$u_{i}$')
    

    plt.figure()
    ns = np.array(ns, dtype=float)
    plt.semilogy(range(len(ns)), ns / ns[0], 'k:', label=r'MG V-cycle')
    plt.axhline(y=1e-12, color='black', linewidth='0.7', alpha=0.8, label=r'$10^{-12}$')
    plt.legend()

    plt.show()
    
    return

def test_pcg():

    f = lambda x, y: 20*np.pi**2 * np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    g = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    u_ex = lambda x, y: np.sin(2*np.pi*x) * np.sin(4*np.pi*y)
    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)

    N          = 2**7
    levels     = 3
    nu1        = 5
    nu2        = 5
    tol        = 1e-13
    max_iter   = 500
    cg_tol     = 1e-3
    cg_maxiter = 500

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])

    u = u_ex(x, y)
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    #u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    u0[1:-1,1:-1] = 0.5
    start = time()
    uh, i, ns = pcg(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    end = time()
    print(f'{(end-start):.2f} s')

    print(f'#iterations = {i}')
    ns = np.array(ns, dtype=float)

    uh_cg, i_cg, ns_cg = my_cg(u0, rhs, N, tol=tol, max_iter=max_iter)
    print(f'#CG-iterations = {i_cg}')

    ns = np.array(ns, dtype=float)
    ns_cg = np.array(ns_cg, dtype=float)
    plt.figure()
    plt.semilogy(range(len(ns)), ns / ns[0], 'k:')
    plt.semilogy(range(len(ns_cg)), ns_cg / ns_cg[0], 'k--')

    plt.show()



    return

def test_pcg_steps():

    f = lambda x, y: 0*x - 1
    g = lambda x, y: np.where(x == 0, 4*y*(1-y), 0*x)
    

    N          = 2**7
    levels     = 7
    nu1        = 4
    nu2        = 4
    tol        = 1e-12
    max_iter   = 1000
    cg_tol     = 1e-12
    cg_maxiter = 1000

    x = np.outer(np.linspace(0, 1, N+1), np.ones(N+1))
    y = np.outer(np.ones(N+1), np.linspace(0, 1, N+1))

    rhs = g(x, y)
    rhs[1:-1, 1:-1] = f(x[1:-1, 1:-1], y[1:-1, 1:-1])
    
    np.random.seed(seed=1)
    u0 = np.copy(rhs)
    u0[1:-1,1:-1] = np.random.random((N-1, N-1))
    
    start = time()
    uh, i, ns, uh_arr = pcg_steps(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    end = time()
    runtime = end - start

    start = time()
    uh_mg, i_mg, ns_mg = mgv_iteration(u0, rhs, N, nu1, nu2, level=1, max_level=levels, tol=tol, max_iter=max_iter, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    end = time()
    runtime_cg = end - start

    print(f'PCG runtime = {runtime*1000:.2f} ms')
    print(f'MG runtime = {runtime_cg*1000:.2f} ms')

    K = len(uh_arr[:5+1])
    s = N // 2**5
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i, uh in enumerate(uh_arr[:5+1]):
        k = uh.shape[0]
        xx = np.outer(np.linspace(0, 1, k), np.ones(k))
        yy = np.outer(np.ones(k), np.linspace(0, 1, k))
        ax = fig.add_subplot(K//2 + K%2, 2, i+1, projection='3d')
        surf = ax.plot_surface(xx[::s,::s], yy[::s,::s], uh[::s,::s],
                            rstride=1, cstride=1, 
                            cmap=matplotlib.cm.viridis, label=f'$u_{i}$')
    

    plt.figure()
    ns = np.array(ns, dtype=float)
    plt.semilogy(range(len(ns)), ns / ns[0], 'k:', label=r'MGCG')
    plt.axhline(y=1e-12, color='black', linewidth='0.7', alpha=0.8, label=r'$10^{-12}$')
    plt.legend()

    plt.show()
    
    return

def main():

    #test_my_cg()

    #test_restriction()

    #test_interpolation()

    #test_residual()

    #test_jacobi()

    #test_mgv()

    #test_mgv_poly()

    #test_mgv_iteration()

    #test_mgv_iteration_steps()

    test_pcg_steps()

    #test_pcg()

if __name__ == '__main__':
    main()
