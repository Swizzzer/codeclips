// 假定m是2^LIMIT-smooth的, 利用MITM的思想遍历获取m
// nvcc -O3 -Xcompiler="-O3 -march=native -fopenmp -pipe" -arch=sm_90 solver.cu -lgmp -o solver
// Usage: "./solver < input.txt"
// input.txt内容为两行, 第一行是十进制的n，第二行是十进制的c
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include <gmpxx.h>

#ifdef _GLIBCXX_PARALLEL
#include <parallel/algorithm>
using __gnu_parallel::sort;
#else
using std::sort;
#endif

using std::uint32_t;
using std::uint64_t;

static constexpr uint32_t LIMIT = (1u << 24);
static constexpr uint32_t BLOCK_SIZE_Y = (1u << 20);
static constexpr unsigned long E65537 = 65537ul;
static constexpr int MAX_LIMBS = 9;

static inline void progress_bar(const char *prefix, uint64_t done,
                                uint64_t total) {
  static const int width = 30;
  double r = total ? (double)done / total : 1.0;
  if (r < 0)
    r = 0;
  if (r > 1)
    r = 1;
  int full = (int)std::round(r * width);
  std::ostringstream oss;
  oss << '\r' << prefix << " [";
  for (int i = 0; i < width; i++)
    oss << (i < full ? '=' : ' ');
  oss << "] " << std::fixed << std::setprecision(1) << (r * 100.0) << "%";
  std::cerr << oss.str();
  if (done == total)
    std::cerr << "\n";
  std::cerr.flush();
}

static bool PY_PRINTABLE[256];
static void init_printable() {
  std::string digits = "0123456789";
  std::string lower = "abcdefghijklmnopqrstuvwxyz";
  std::string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  std::string ws = " \t\n\r\v\f";
  std::string all = digits + lower + upper + punct + ws;
  std::fill(std::begin(PY_PRINTABLE), std::end(PY_PRINTABLE), false);
  for (unsigned char ch : all)
    PY_PRINTABLE[ch] = true;
}
static inline std::string decode_m(uint64_t m48) {
  char s[6];
  for (int i = 5; i >= 0; --i) {
    s[i] = char(m48 & 0xFFu);
    m48 >>= 8;
  }
  return std::string(s, 6);
}
static inline bool is_py_printable_6(const std::string &s) {
  if (s.size() != 6)
    return false;
  for (unsigned char ch : s) {
    if (ch > 0x7F || !PY_PRINTABLE[ch])
      return false;
  }
  return true;
}
static void mpz_to_limbs_le(const mpz_t in, int L, std::vector<uint64_t> &out) {
  out.assign(L, 0);
  size_t count = 0;
  mpz_export(out.data(), &count, -1, sizeof(uint64_t), 0, 0, in);
}
static void limbs_to_mpz_le(const uint64_t *in, int L, mpz_t out) {
  mpz_import(out, L, -1, sizeof(uint64_t), 0, 0, in);
}

static uint64_t inv64(uint64_t n0) {
  uint64_t x = 1;
  for (int i = 0; i < 6; i++) {
    __uint128_t t = (__uint128_t)x * (2 - (__uint128_t)n0 * x);
    x = (uint64_t)t;
  }
  return ~x + 1;
}

__constant__ uint64_t d_mod[MAX_LIMBS];
__constant__ uint64_t d_R2[MAX_LIMBS];
__constant__ uint64_t d_n0inv;
__constant__ int d_L;

struct Big {
  uint64_t v[MAX_LIMBS];
};

__device__ __forceinline__ void big_zero(Big &a) {
#pragma unroll
  for (int i = 0; i < MAX_LIMBS; i++)
    a.v[i] = 0ull;
}
__device__ __forceinline__ void big_copy(const Big &a, Big &b) {
#pragma unroll
  for (int i = 0; i < MAX_LIMBS; i++)
    b.v[i] = a.v[i];
}
__device__ __forceinline__ void big_from_u32(uint32_t x, Big &a) {
  a.v[0] = (uint64_t)x;
#pragma unroll
  for (int i = 1; i < MAX_LIMBS; i++)
    a.v[i] = 0ull;
}
__device__ __forceinline__ bool ge_mod(const Big &t) {
  for (int i = d_L - 1; i >= 0; --i) {
    if (t.v[i] > d_mod[i])
      return true;
    if (t.v[i] < d_mod[i])
      return false;
  }
  return true;
}
__device__ __forceinline__ void sub_mod(Big &t) {
  unsigned __int128 borrow = 0;
  for (int i = 0; i < d_L; i++) {
    unsigned __int128 cur =
        (unsigned __int128)t.v[i] - d_mod[i] - (uint64_t)borrow;
    t.v[i] = (uint64_t)cur;
    borrow = (cur >> 64) & 1;
  }
}
__device__ void mont_mul(const Big &a, const Big &b, Big &res) {
  Big t;
  big_zero(t);
  uint64_t t_hi = 0ull;
  for (int i = 0; i < d_L; i++) {
    unsigned __int128 carry = 0;
    for (int j = 0; j < d_L; j++) {
      unsigned __int128 prod = (unsigned __int128)a.v[j] * b.v[i];
      unsigned __int128 sum = (unsigned __int128)t.v[j] + prod + carry;
      t.v[j] = (uint64_t)sum;
      carry = sum >> 64;
    }
    unsigned __int128 sumhi = (unsigned __int128)t_hi + carry;
    t_hi = (uint64_t)sumhi;

    uint64_t m = (uint64_t)((unsigned __int128)t.v[0] * d_n0inv);

    unsigned __int128 carry2 = 0;
    for (int j = 0; j < d_L; j++) {
      unsigned __int128 sum2 =
          (unsigned __int128)t.v[j] + (unsigned __int128)m * d_mod[j] + carry2;
      t.v[j] = (uint64_t)sum2;
      carry2 = sum2 >> 64;
    }
    unsigned __int128 sumhi2 = (unsigned __int128)t_hi + carry2;
    t_hi = (uint64_t)sumhi2;

    for (int k = 0; k < d_L - 1; k++)
      t.v[k] = t.v[k + 1];
    t.v[d_L - 1] = t_hi;
    t_hi = 0ull;
  }
  if (ge_mod(t))
    sub_mod(t);
  big_copy(t, res);
}
__device__ __forceinline__ void to_mont(const Big &a, Big &aMont) {
  Big R2;
#pragma unroll
  for (int i = 0; i < MAX_LIMBS; i++)
    R2.v[i] = (i < d_L ? d_R2[i] : 0ull);
  mont_mul(a, R2, aMont);
}
__device__ __forceinline__ void from_mont(const Big &aMont, Big &a) {
  Big one_plain;
  big_zero(one_plain);
  one_plain.v[0] = 1;
  mont_mul(aMont, one_plain, a);
}
__device__ void pow65537_u32(uint32_t base, Big &out_normal) {
  Big a, aMont, t, res;
  big_from_u32(base, a);
  to_mont(a, aMont);
  big_copy(aMont, t);
#pragma unroll 16
  for (int i = 0; i < 16; i++)
    mont_mul(t, t, t);
  mont_mul(t, aMont, res);
  from_mont(res, out_normal);
}

__global__ void kernel_x_keys(uint64_t *out_keys, uint32_t global_off,
                              uint32_t count) {
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  uint64_t idx = global_off + gid;
  uint32_t x = (uint32_t)idx + 1;
  Big r;
  pow65537_u32(x, r);
  out_keys[idx] = r.v[0];
}

__global__ void kernel_y_full(uint64_t *out_limbs, uint32_t y_start,
                              uint32_t B) {
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B)
    return;
  uint32_t y = y_start + (uint32_t)gid;
  Big r;
  pow65537_u32(y, r);
  uint64_t *p = out_limbs + gid * d_L;
#pragma unroll
  for (int i = 0; i < MAX_LIMBS; i++)
    if (i < d_L)
      p[i] = r.v[i];
}

struct Finger {
  uint64_t key;
  uint32_t x;
};
static inline bool operator<(const Finger &a, const Finger &b) {
  return a.key < b.key || (a.key == b.key && a.x < b.x);
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  init_printable();

  std::string ns, cs;
  if (!(std::cin >> ns >> cs)) {
    std::cerr << "Usage: <n> <c>\n";
    return 1;
  }

  mpz_t N, C;
  mpz_inits(N, C, nullptr);
  if (mpz_set_str(N, ns.c_str(), 10) != 0 ||
      mpz_set_str(C, cs.c_str(), 10) != 0) {
    std::cerr << "Invalid decimal input.\n";
    return 1;
  }

  int nbits = mpz_sizeinbase(N, 2);
  int L = (nbits + 63) / 64;
  if (L > MAX_LIMBS) {
    std::cerr << "[!] Modulus too large (> " << (64 * MAX_LIMBS) << " bits)\n";
    return 1;
  }

  std::vector<uint64_t> mod_le, R2_le;
  mpz_t R, R2;
  mpz_inits(R, R2, nullptr);
  mpz_set_ui(R, 0);
  mpz_setbit(R, 64 * L);
  mpz_mod(R, R, N);
  mpz_mul(R2, R, R);
  mpz_mod(R2, R2, N);
  mpz_to_limbs_le(N, L, mod_le);
  mpz_to_limbs_le(R2, L, R2_le);
  uint64_t n0 = mod_le[0];
  uint64_t n0inv = inv64(n0);

  cudaMemcpyToSymbol(d_mod, mod_le.data(), sizeof(uint64_t) * L);
  cudaMemcpyToSymbol(d_R2, R2_le.data(), sizeof(uint64_t) * L);
  cudaMemcpyToSymbol(d_n0inv, &n0inv, sizeof(uint64_t));
  cudaMemcpyToSymbol(d_L, &L, sizeof(int));

  const uint32_t XN = LIMIT - 1;
  std::cerr << "[*] Computing x^e mod n on GPU...\n";

  uint64_t *d_keys = nullptr;
  cudaMalloc(&d_keys, sizeof(uint64_t) * XN);

  const int TPB = 256;
  const uint32_t CHUNK = 4u << 20;
  for (uint32_t off = 0; off < XN; off += CHUNK) {
    uint32_t count = std::min<uint32_t>(CHUNK, XN - off);
    uint32_t b = (count + TPB - 1) / TPB;
    kernel_x_keys<<<b, TPB>>>(d_keys, off, count);
    cudaDeviceSynchronize();
    progress_bar("  - x^e GPU", off + count, XN);
  }

  std::vector<uint64_t> keys_host(XN);
  cudaMemcpy(keys_host.data(), d_keys, sizeof(uint64_t) * XN,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys);

  std::vector<Finger> table(XN);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < (int64_t)XN; i++) {
    table[i].key = keys_host[i];
    table[i].x = (uint32_t)i + 1;
    if ((i & ((1 << 20) - 1)) == 0)
      progress_bar("  - build table", i, XN);
  }
  progress_bar("  - build table", XN, XN);

  std::cerr << "[*] Sorting on CPU...\n";
  sort(table.begin(), table.end());

  std::cerr << "[*] Scanning y side...\n";

  uint64_t *d_ybuf = nullptr;
  cudaMalloc(&d_ybuf, sizeof(uint64_t) * (size_t)BLOCK_SIZE_Y * L);
  uint64_t *h_ybuf = nullptr;
  cudaMallocHost(&h_ybuf, sizeof(uint64_t) * (size_t)BLOCK_SIZE_Y * L);

  std::vector<mpz_class> Y(BLOCK_SIZE_Y), Pref(BLOCK_SIZE_Y);

  uint64_t processed = 0;
  for (uint32_t y0 = 1; y0 <= XN; y0 += BLOCK_SIZE_Y) {
    uint32_t B = std::min<uint32_t>(XN - (y0 - 1), BLOCK_SIZE_Y);

    const uint32_t b = (B + TPB - 1) / TPB;
    kernel_y_full<<<b, TPB>>>(d_ybuf, y0, B);
    cudaDeviceSynchronize();
    cudaMemcpy(h_ybuf, d_ybuf, sizeof(uint64_t) * (size_t)B * L,
               cudaMemcpyDeviceToHost);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)B; i++) {
      limbs_to_mpz_le(h_ybuf + (size_t)i * L, L, Y[i].get_mpz_t());
    }

    mpz_set(Pref[0].get_mpz_t(), Y[0].get_mpz_t());
    for (uint32_t i = 1; i < B; i++) {
      mpz_mul(Pref[i].get_mpz_t(), Pref[i - 1].get_mpz_t(), Y[i].get_mpz_t());
      mpz_mod(Pref[i].get_mpz_t(), Pref[i].get_mpz_t(), N);
    }

    mpz_t P, invP, tmp, acc;
    mpz_inits(P, invP, tmp, acc, nullptr);
    mpz_set(P, Pref[B - 1].get_mpz_t());
    if (mpz_invert(invP, P, N) == 0) {
      mpz_clears(P, invP, tmp, acc, nullptr);
      processed += B;
      progress_bar("  - scan y", processed, (uint64_t)XN);
      continue;
    }
    mpz_set(acc, invP);

    for (int i = (int)B - 1; i >= 0; --i) {
      if (i > 0) {
        mpz_mul(tmp, Pref[i - 1].get_mpz_t(), acc);
        mpz_mod(tmp, tmp, N);
      } else {
        mpz_set_ui(tmp, 1);
        mpz_mul(tmp, tmp, acc);
        mpz_mod(tmp, tmp, N);
      }
      mpz_mul(tmp, C, tmp);
      mpz_mod(tmp, tmp, N);
      uint64_t k = mpz_get_ui(tmp);

      Finger ql{k, 0};
      auto it = std::lower_bound(table.begin(), table.end(), ql);
      for (; it != table.end() && it->key == k; ++it) {
        uint32_t x = it->x;
        mpz_t xe, lhs;
        mpz_inits(xe, lhs, nullptr);
        mpz_set_ui(xe, x);
        mpz_powm_ui(xe, xe, E65537, N);
        mpz_mul(lhs, xe, Y[i].get_mpz_t());
        mpz_mod(lhs, lhs, N);
        if (mpz_cmp(lhs, C) == 0) {
          uint64_t y = (uint64_t)y0 + (uint32_t)i;
          uint64_t m48 = (uint64_t)x * (uint64_t)y;
          std::string pin = decode_m(m48);
          if (is_py_printable_6(pin)) {
            std::cerr << "\n[*] HIT! x=" << x << ", y=" << y << "\n";
            std::cout << pin << "\n";
            mpz_clears(xe, lhs, P, invP, tmp, acc, nullptr);
            cudaFree(d_ybuf);
            cudaFreeHost(h_ybuf);
            mpz_clears(N, C, R, R2, nullptr);
            return 0;
          }
        }
        mpz_clears(xe, lhs, nullptr);
      }
      mpz_mul(acc, acc, Y[i].get_mpz_t());
      mpz_mod(acc, acc, N);

      if ((((uint64_t)y0 - 1) + (uint32_t)(B - i)) % (1u << 18) == 0) {
        uint64_t done = ((uint64_t)y0 - 1) + (uint32_t)(B - i);
        progress_bar("  - scan y", done, (uint64_t)XN);
      }
    }

    mpz_clears(P, invP, tmp, acc, nullptr);
    processed += B;
    progress_bar("  - scan y", processed, (uint64_t)XN);
  }

  std::cerr << "\n[!] Not found under 2^24-smooth constraint.\n";
  cudaFree(d_ybuf);
  cudaFreeHost(h_ybuf);
  mpz_clears(N, C, R, R2, nullptr);
  return 2;
}
