use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_ffn_gate_kernel::{
    M, K, N, pilot_golden_ffn_gate_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}

fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
    data.iter().map(|v| f32::from(*v)).collect()
}

#[tokio::main]
async fn main() { println!("2D tiling test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tiling_2d() {
        let mut ctx = Context::acquire();

        // K=128, N=1024 → K=64×2, N=512×2 = 4 tiles
        let x0_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/tiling_2d_x0.bin"))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let x1_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/tiling_2d_x1.bin"))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;

        let mut n_partials: Vec<Vec<f32>> = Vec::new();

        for ni in 0..2u32 {
            let mut partial: Vec<f32> = vec![0.0; 512 * 512];

            for ki in 0..2u32 {
                let w_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
                    load_bf16_bin(&format!("data/tiling_2d_w_n{ni}_k{ki}.bin")))
                    .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;

                let x_hbm = if ki == 0 { &x0_hbm } else { &x1_hbm };
                let out_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, x_hbm, &w_hbm)).await;
                let tile: Vec<bf16> = out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
                let tile_f32 = bf16_to_f32(&tile);

                for (p, t) in partial.iter_mut().zip(tile_f32.iter()) {
                    *p += t;
                }
            }
            n_partials.push(partial);
        }

        // row-major 재배열: [M, N_2D] = [512, 1024]
        let m_dim = 512usize;
        let n_tile = 512usize;
        let n_tiles = 2usize;
        let mut out_all = vec![0.0f32; m_dim * n_tile * n_tiles];
        for t in 0..n_tiles {
            for r in 0..m_dim {
                for c in 0..n_tile {
                    out_all[r * (n_tile * n_tiles) + t * n_tile + c] = n_partials[t][r * n_tile + c];
                }
            }
        }

        let exp_bytes = load_bf16_bin("data/tiling_2d_expected.bin");
        let expected: Vec<bf16> = exp_bytes.chunks(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let exp_f32 = bf16_to_f32(&expected);

        println!("=== 2D tiling (K=128 N=1024, 2x2 tiles) ===");
        for i in 0..4 {
            let diff = (exp_f32[i] - out_all[i]).abs();
            println!("  [{i}] expected={:.4}  actual={:.4}  diff={:.4}", exp_f32[i], out_all[i], diff);
        }

        let mut fail = 0;
        for (&e, &a) in exp_f32.iter().zip(out_all.iter()) {
            let diff = (e - a).abs();
            let tol = (e.abs() * 0.1).max(2.0);
            if diff > tol { fail += 1; }
        }
        assert!(fail == 0, "2D tiling mismatch: {fail}/{}", exp_f32.len());
        println!("PASS: {} values matched (2x2 tiles)", out_all.len());
    }
}
