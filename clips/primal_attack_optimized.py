# Copied from 糖醋小鸡块
# https://tangcuxiaojikuai.xyz/post/758dd33a.html
from sage.all import matrix, block_matrix, Zmod, ZZ, diagonal_matrix, vector, GF


def primal_attack(A, b, m, n, p, esz):
    L = block_matrix(
        [
            [matrix(Zmod(p), A).T.echelon_form().change_ring(ZZ), 0],
            [matrix.zero(m - n, n).augment(matrix.identity(m - n) * p), 0],
            [matrix(ZZ, b), 1],
        ]
    )
    # print(L.dimensions())
    Q = diagonal_matrix([1] * m + [esz])
    L *= Q
    L = L.LLL()
    L /= Q
    res = L[0]
    if res[-1] == 1:
        e = vector(GF(p), res[:m])
    elif res[-1] == -1:
        e = -vector(GF(p), res[:m])
    s = matrix(Zmod(p), A).solve_right((vector(Zmod(p), b) - e))
    return s
