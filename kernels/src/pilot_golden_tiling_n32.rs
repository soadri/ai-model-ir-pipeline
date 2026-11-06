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
async fn main() { println!("N-tiling test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tiling_n() {
        let mut ctx = Context::acquire();

        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/tiling_n_x.bin"))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;

        let mut tiles: Vec<Vec<f32>> = Vec::new();

        for i in 0..4u32 {
            let w_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
                load_bf16_bin(&format!("data/tiling_n_w{i}.bin")))
                .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;

            let out_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &w_hbm)).await;
            let tile: Vec<bf16> = out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
            tiles.push(bf16_to_f32(&tile));
        }

        // row-major 재배열: [M, N_TOTAL] = [512, 2048]
        let m_dim = 512usize;
        let n_tile = 512usize;
        let n_tiles = 4usize;
        let mut out_all = vec![0.0f32; m_dim * n_tile * n_tiles];
        for t in 0..n_tiles {
            for r in 0..m_dim {
                for c in 0..n_tile {
                    out_all[r * (n_tile * n_tiles) + t * n_tile + c] = tiles[t][r * n_tile + c];
                }
            }
        }

        let exp_bytes = load_bf16_bin("data/tiling_n_expected.bin");
        let expected: Vec<bf16> = exp_bytes.chunks(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let exp_f32 = bf16_to_f32(&expected);

        println!("=== N-tiling (N=2048, tile=512, 4 tiles) ===");
        for i in 0..4 {
            let diff = (exp_f32[i] - out_all[i]).abs();
            println!("  [{i}] expected={:.4}  actual={:.4}  diff={:.4}", exp_f32[i], out_all[i], diff);
        }

        let mut fail = 0;
        for (idx, (&e, &a)) in exp_f32.iter().zip(out_all.iter()).enumerate() {
            let diff = (e - a).abs();
            let tol = (e.abs() * 0.05).max(1.0);
            if diff > tol { fail += 1; }
        }
        assert!(fail == 0, "N-tiling mismatch: {fail}/{}", exp_f32.len());
        println!("PASS: {} values matched (4 tiles)", out_all.len());
    }
}
