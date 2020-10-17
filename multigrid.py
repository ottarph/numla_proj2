import numpy as np

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


def main():

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


if __name__ == '__main__':
    main()
