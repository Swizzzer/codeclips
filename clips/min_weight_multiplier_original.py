import heapq
from tqdm import trange


def min_hamming_mul(n: int) -> tuple[int, int, str]:
    pq = [(1, 1, "1", 1 % n)]
    best = {}

    while pq:
        w, L, s, r = heapq.heappop(pq)
        if r == 0:
            t = int(s, 2)
            c = t // n
            return c, t, s

        if r in best and best[r] <= (w, L):
            continue
        best[r] = (w, L)

        r0 = (r * 2) % n
        heapq.heappush(pq, (w, L + 1, s + "0", r0))

        r1 = (r * 2 + 1) % n
        heapq.heappush(pq, (w + 1, L + 1, s + "1", r1))


if __name__ == "__main__":
    res = []
    for n in trange(1, 2**8):
        c, t, b = min_hamming_mul(n)
        res.append(c)
    print(",".join(map(str, res)))