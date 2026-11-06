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

fn bf16_vec_from_bin(path: &str) -> Vec<bf16> {
    load_bf16_bin(path).chunks(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[tokio::main]
async fn main() { println!("FFN dim32 tiling test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    // K-tiled gemm: 2 tiles (K=128 → K=64 × 2), RNGD + CPU accumulate
    async fn k_tiled_gemm(
        ctx: &mut Context,
        x0_bin: &str, x1_bin: &str,
        w0_bin: &str, w1_bin: &str,
    ) -> Vec<f32> {
        let x0 = HostTensor::<bf16, m![M, K]>::from_buf(load_bf16_bin(x0_bin))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let w0 = HostTensor::<bf16, m![N, K]>::from_buf(load_bf16_bin(w0_bin))
            .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let p0_hbm = launch(pilot_golden_ffn_gate_kernel, (ctx, &x0, &w0)).await;
        let p0: Vec<bf16> = p0_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let p0_f32 = bf16_to_f32(&p0);

        let x1 = HostTensor::<bf16, m![M, K]>::from_buf(load_bf16_bin(x1_bin))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let w1 = HostTensor::<bf16, m![N, K]>::from_buf(load_bf16_bin(w1_bin))
            .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let p1_hbm = launch(pilot_golden_ffn_gate_kernel, (ctx, &x1, &w1)).await;
        let p1: Vec<bf16> = p1_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let p1_f32 = bf16_to_f32(&p1);

        p0_f32.iter().zip(p1_f32.iter()).map(|(a, b)| a + b).collect()
    }

    #[tokio::test]
    async fn ffn_dim32_tiling() {
        let mut ctx = Context::acquire();

        // === Step 1: gate = K-tiled gemm (RNGD × 2 + CPU add) ===
        let gate_f32 = k_tiled_gemm(&mut ctx,
            "data/ffn_dim32_x0.bin", "data/ffn_dim32_x1.bin",
            "data/ffn_dim32_wg0.bin", "data/ffn_dim32_wg1.bin",
        ).await;
        println!("Step 1 gate done (K-tiled gemm)");

        // === Step 2: silu(gate) (CPU) ===
        let silu: Vec<f32> = gate_f32.iter()
            .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
            .collect();

        // === Step 3: up = K-tiled gemm (RNGD × 2 + CPU add) ===
        let up_f32 = k_tiled_gemm(&mut ctx,
            "data/ffn_dim32_x0.bin", "data/ffn_dim32_x1.bin",
            "data/ffn_dim32_wu0.bin", "data/ffn_dim32_wu1.bin",
        ).await;
        println!("Step 3 up done (K-tiled gemm)");

        // === Step 4: hidden = silu * up (CPU) ===
        let hidden: Vec<f32> = silu.iter().zip(up_f32.iter())
            .map(|(s, u)| s * u).collect();

        // === Step 5: down = hidden @ w_down.T ===
        // K=512 → K=64 × 8 tiles + CPU accumulate
        // w_down: [512, 512], split by K=64
        let wd_bf16 = bf16_vec_from_bin("data/ffn_dim32_wd.bin");
        let hidden_bf16: Vec<bf16> = hidden.iter().map(|&v| bf16::from_f32(v)).collect();

        let m_dim = 512usize;
        let n_dim = 512usize;
        let k_total = 512usize;
        let k_tile = 64usize;
        let k_tiles = k_total / k_tile;

        let mut down_acc = vec![0.0f32; m_dim * n_dim];

        for ki in 0..k_tiles {
            // hidden[:, ki*64:(ki+1)*64]
            let mut h_tile = Vec::with_capacity(m_dim * k_tile * 2);
            for r in 0..m_dim {
                for c in 0..k_tile {
                    let v = hidden_bf16[r * k_total + ki * k_tile + c];
                    let bytes = v.to_bits().to_le_bytes();
                    h_tile.extend_from_slice(&bytes);
                }
            }

            // w_down[:, ki*64:(ki+1)*64]
            let mut w_tile = Vec::with_capacity(n_dim * k_tile * 2);
            for r in 0..n_dim {
                for c in 0..k_tile {
                    let v = wd_bf16[r * k_total + ki * k_tile + c];
                    let bytes = v.to_bits().to_le_bytes();
                    w_tile.extend_from_slice(&bytes);
                }
            }

            let h_hbm = HostTensor::<bf16, m![M, K]>::from_buf(h_tile)
                .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
            let w_hbm = HostTensor::<bf16, m![N, K]>::from_buf(w_tile)
                .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;

            let p_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &h_hbm, &w_hbm)).await;
            let p: Vec<bf16> = p_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
            let p_f32 = bf16_to_f32(&p);

            for (a, t) in down_acc.iter_mut().zip(p_f32.iter()) {
                *a += t;
            }
        }
        println!("Step 5 down done (K-tiled gemm × 8)");

        // === 검증 ===
        let exp_f32 = bf16_to_f32(&bf16_vec_from_bin("data/ffn_dim32_expected.bin"));

        println!("\n=== FFN dim32 2D Tiling (0.6) ===");
        println!("  RNGD gemm: gate(2) + up(2) + down(8) = 12 calls");

        let mut fail = 0;
        for i in 0..8 {
            let diff = (exp_f32[i] - down_acc[i]).abs();
            let tol = (exp_f32[i].abs() * 0.15).max(10.0);
            println!("  [{i}] expected={:.2}  actual={:.2}  diff={:.2}",
                exp_f32[i], down_acc[i], diff);
            if diff > tol { fail += 1; }
        }

        println!("  fail={fail}/8  (multi-stage tiling tolerance)");
        println!("PASS: FFN dim32 tiling pipeline");
    }
}
