// [package]
// name = "min_weight_multiplier"
// version = "0.1.0"
// edition = "2021"

// [dependencies]
// num-bigint = "0.4"
// num-traits = "0.2"
// indicatif   = "0.17"
use indicatif::{ProgressBar, ProgressStyle};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::VecDeque;

#[derive(Clone)]
struct State {
    weight: u32,
    len: u32,
    bits: Vec<u8>,
}

impl State {
    fn better_than(&self, other: &State) -> bool {
        self.weight < other.weight
            || (self.weight == other.weight
                && (self.len < other.len || (self.len == other.len && self.bits < other.bits)))
    }
}
/// a function to find the multiplier t of given n, which makes the hamming weight of t*n smallest
/// used as the solver of `Ahoo` in CryptoCTF 2024
fn min_weight_multiplier(n: usize) -> (BigUint, BigUint, String) {
    let start = 1 % n;

    let mut best: Vec<Option<State>> = vec![None; n];
    let mut q: VecDeque<usize> = VecDeque::new();

    best[start] = Some(State {
        weight: 1,
        len: 1,
        bits: vec![1],
    });
    q.push_front(start);

    while let Some(r) = q.pop_front() {
        let cur = best[r].clone().unwrap();

        if r == 0 {
            let bin_string: String = cur.bits.iter().map(|b| (b'0' + *b) as char).collect();

            let mut t = BigUint::zero();
            for &b in &cur.bits {
                t <<= 1usize;
                if b == 1 {
                    t += BigUint::one();
                }
            }
            let c = &t / n;
            return (c, t, bin_string);
        }

        // 0-1 BFS：先 cost=0 的边 push_front，再 cost=1 push_back
        for &(bit, cost) in &[(0u8, 0u32), (1u8, 1u32)] {
            let nxt_r = (r * 2 + bit as usize) % n;

            let mut bits = cur.bits.clone();
            bits.push(bit);
            let cand = State {
                weight: cur.weight + cost,
                len: cur.len + 1,
                bits,
            };

            // 如果更优，就替换并入队
            if best[nxt_r]
                .as_ref()
                .map_or(true, |old| cand.better_than(old))
            {
                best[nxt_r] = Some(cand);
                if cost == 0 {
                    q.push_front(nxt_r)
                } else {
                    q.push_back(nxt_r)
                }
            }
        }
    }

    unreachable!("search exhausted without hitting remainder 0");
}

fn main() {
    let max_n = 1usize << 8;
    let bar = ProgressBar::new((max_n - 1) as u64).with_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {eta_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let mut results = Vec::with_capacity(max_n - 1);
    for n in 1..max_n {
        let (c, _, _) = min_weight_multiplier(n);
        results.push(c.to_string());
        bar.inc(1);
    }
    bar.finish_with_message("done!");

    println!("{}", results.join(","));
}
