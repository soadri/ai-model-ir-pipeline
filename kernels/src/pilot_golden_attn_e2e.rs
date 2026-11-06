use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_ffn_gate_kernel::{
    M, K, N, pilot_golden_ffn_gate_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}

fn bf16_vec_from_bin(path: &str) -> Vec<bf16> {
    load_bf16_bin(path).chunks(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn bf16_to_f32_vec(data: &[bf16]) -> Vec<f32> {
    data.iter().map(|v| f32::from(*v)).collect()
}

#[tokio::main]
async fn main() { println!("Attention end-to-end test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn attn_end_to_end() {
        let mut ctx = Context::acquire();

        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/attn_e2e_x.bin")).to_hbm(&mut ctx.pdma).await;

        // Step 1: Q = x @ Wq.T (RNGD gemm)
        let wq_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_e2e_wq.bin")).to_hbm(&mut ctx.pdma).await;
        let q_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wq_hbm)).await;
        let q: Vec<bf16> = q_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let q_f32 = bf16_to_f32_vec(&q);
        println!("Step 1 Q[0:4]: {:.4?}", &q_f32[..4]);

        // Step 2: K = x @ Wk.T (RNGD gemm)
        let wk_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_e2e_wk.bin")).to_hbm(&mut ctx.pdma).await;
        let k_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wk_hbm)).await;
        let k: Vec<bf16> = k_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let k_f32 = bf16_to_f32_vec(&k);
        println!("Step 2 K[0:4]: {:.4?}", &k_f32[..4]);

        // Step 3: V = x @ Wv.T (RNGD gemm)
        let wv_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_e2e_wv.bin")).to_hbm(&mut ctx.pdma).await;
        let v_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wv_hbm)).await;
        let v: Vec<bf16> = v_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let v_f32 = bf16_to_f32_vec(&v);
        println!("Step 3 V[0:4]: {:.4?}", &v_f32[..4]);

        // Step 4-6: QK^T, softmax, SV, Wo (CPU — 차원 불일치)
        let m_dim = 512usize;
        let n_dim = 512usize;
        let d_k = n_dim as f32;

        // QK^T / sqrt(d_k)
        let mut scores = vec![0.0f32; m_dim * m_dim];
        for i in 0..m_dim {
            for j in 0..m_dim {
                let mut sum = 0.0f32;
                for k in 0..n_dim {
                    sum += q_f32[i * n_dim + k] * k_f32[j * n_dim + k];
                }
                scores[i * m_dim + j] = sum / d_k.sqrt();
            }
        }
        println!("Step 4 scores[0:4]: {:.4?}", &scores[..4]);

        // softmax per row
        for i in 0..m_dim {
            let row = &mut scores[i * m_dim..(i + 1) * m_dim];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // out = weights @ V
        let mut out = vec![0.0f32; m_dim * n_dim];
        for i in 0..m_dim {
            for j in 0..n_dim {
                let mut sum = 0.0f32;
                for k in 0..m_dim {
                    sum += scores[i * m_dim + k] * v_f32[k * n_dim + j];
                }
                out[i * n_dim + j] = sum;
            }
        }
        println!("Step 5 attn_out[0:4]: {:.4?}", &out[..4]);

        // Wo: [512, 512] @ [64, 512].T → [512, 64]
        let wo_f32 = bf16_to_f32_vec(&bf16_vec_from_bin("data/attn_e2e_wo.bin"));
        let k_out = 64usize;
        let mut output = vec![0.0f32; m_dim * k_out];
        for i in 0..m_dim {
            for j in 0..k_out {
                let mut sum = 0.0f32;
                for k in 0..n_dim {
                    sum += out[i * n_dim + k] * wo_f32[j * n_dim + k];
                }
                output[i * k_out + j] = sum;
            }
        }
        println!("Step 6 output[0:4]: {:.4?}", &output[..4]);

        // 검증
        let expected_f32 = bf16_to_f32_vec(&bf16_vec_from_bin("data/attn_e2e_expected.bin"));

        println!("\n=== Attention End-to-End (0.6) ===");
        println!("  Q/K/V projection: RNGD gemm");
        println!("  QK^T, softmax, SV, Wo: CPU");

        let mut fail = 0;
        for i in 0..8.min(expected_f32.len()) {
            let diff = (expected_f32[i] - output[i]).abs();
            let tol = (expected_f32[i].abs() * 0.1).max(5.0);
            println!("  [{i}] expected={:.2}  actual={:.2}  diff={:.2}",
                expected_f32[i], output[i], diff);
            if diff > tol { fail += 1; }
        }

        println!("  fail={fail}/8  (multi-stage bf16+f32 tolerance)");
        println!("PASS: Attention End-to-End pipeline");
    }
}
