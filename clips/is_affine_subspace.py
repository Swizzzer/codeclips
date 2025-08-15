from sage.all import *
import ast


def analyze_vecs(vecs):
    """
    分析 GF(2) 上的一组向量（用二进制字符串或 0/1 列表表示）
    输出：
    - 向量空间维度
    - 是否为完整仿射子空间
    - 基向量
    """

    if not vecs:
        raise ValueError("vecs 不能为空")

    L = len(vecs[0])
    F = GF(2)
    V = VectorSpace(F, L)

    if isinstance(vecs[0], str):
        vectors = [V(list(map(int, v))) for v in vecs]
    else:
        vectors = [V(v) for v in vecs]

    M = matrix(F, vectors)

    # 取第一个向量, 如果给出的数据经过shuffle则需要检查所有向量
    v0 = vectors[0]
    M_shift = M + matrix(F, [v0 for _ in range(M.nrows())])
    dim = M_shift.rank()
    is_affine = len(vectors) == 2**dim
    basis = M_shift.row_space().basis()

    return {"dim": dim, "is_affine": is_affine, "basis": basis}


if __name__ == "__main__":
    vecs = ast.literal_eval(open("output.txt").readline().strip())
    result = analyze_vecs(vecs)
    print(f"维度: {result['dim']}")
    print(f"是否是仿射子空间: {result['is_affine']}")
    print("基向量:")
    for b in result["basis"]:
        print(b)
