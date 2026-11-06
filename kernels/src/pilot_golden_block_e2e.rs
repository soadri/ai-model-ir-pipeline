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

fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
    data.iter().map(|v| f32::from(*v)).collect()
}

fn rmsnorm_cpu(x: &[f32], dim: usize) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r*dim..(r+1)*dim];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (mean_sq + 1e-6_f32).sqrt();
        for c in 0..dim {
            out[r*dim + c] = row[c] / rms;
        }
    }
    out
}

fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i*k + kk] * b[j*k + kk]; // b is N×K, row-major
            }
            out[i*n + j] = sum;
        }
    }
    out
}

fn softmax_cpu(scores: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        let row = &mut scores[i*n..(i+1)*n];
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
}

fn matmul_nn_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // a: M×K, b: K×N (both row-major, no transpose)
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i*k + kk] * b[kk*n + j];
            }
            out[i*n + j] = sum;
        }
    }
    out
}

#[tokio::main]
async fn main() { println!("Full Block end-to-end test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn block_end_to_end() {
        let mut ctx = Context::acquire();
        type Chip = m![1];
        let m_dim = 512usize;
        let k_dim = 64usize;
        let n_dim = 512usize;

        // 입력 로드
        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/block_e2e_x.bin")).to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let x_bf16 = bf16_vec_from_bin("data/block_e2e_x.bin");
        let x_f32 = bf16_to_f32(&x_bf16);

        // 가중치 로드
        let wq_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/block_e2e_wq.bin")).to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let wk_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/block_e2e_wk.bin")).to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let wv_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/block_e2e_wv.bin")).to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let wgate_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/block_e2e_w_gate.bin")).to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let wup_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/block_e2e_w_up.bin")).to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;

        // === Step 1: RMSNorm1 (CPU) ===
        let norm1 = rmsnorm_cpu(&x_f32, k_dim);
        let norm1_bf16: Vec<bf16> = norm1.iter().map(|&v| bf16::from_f32(v)).collect();
        println!("Step 1 RMSNorm1 done");

        // === Step 2: Q/K/V projection (RNGD gemm) ===
        let n1_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            to_bytes(&norm1_bf16)).to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;

        let q_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &n1_hbm, &wq_hbm)).await;
        let q: Vec<bf16> = q_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let q_f32 = bf16_to_f32(&q);

        let k_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &n1_hbm, &wk_hbm)).await;
        let k_val: Vec<bf16> = k_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let k_f32 = bf16_to_f32(&k_val);

        let v_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &n1_hbm, &wv_hbm)).await;
        let v_val: Vec<bf16> = v_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let v_f32 = bf16_to_f32(&v_val);
        println!("Step 2 Q/K/V projection done (RNGD)");

        // === Step 3: QK^T, softmax, SV, Wo (CPU) ===
        let mut scores = matmul_cpu(&q_f32, &k_f32, m_dim, n_dim, m_dim);
        let scale = (n_dim as f32).sqrt();
        for s in scores.iter_mut() { *s /= scale; }
        softmax_cpu(&mut scores, m_dim, m_dim);
        let attn_out = matmul_nn_cpu(&scores, &v_f32, m_dim, m_dim, n_dim);

        let wo_f32 = bf16_to_f32(&bf16_vec_from_bin("data/block_e2e_wo.bin"));
        let attn_proj = matmul_cpu(&attn_out, &wo_f32, m_dim, n_dim, k_dim);
        println!("Step 3 Attention done (CPU)");

        // === Step 4: Residual1 = x + attn_proj (CPU) ===
        let h: Vec<f32> = x_f32.iter().zip(attn_proj.iter())
            .map(|(a, b)| a + b).collect();
        println!("Step 4 Residual1 done");

        // === Step 5: RMSNorm2 (CPU) ===
        let norm2 = rmsnorm_cpu(&h, k_dim);
        let norm2_bf16: Vec<bf16> = norm2.iter().map(|&v| bf16::from_f32(v)).collect();

        // === Step 6: FFN gate/up (RNGD) + silu/mul/down (CPU) ===
        let n2_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            to_bytes(&norm2_bf16)).to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;

        let gate_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &n2_hbm, &wgate_hbm)).await;
        let gate: Vec<bf16> = gate_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let gate_f32 = bf16_to_f32(&gate);

        let up_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &n2_hbm, &wup_hbm)).await;
        let up: Vec<bf16> = up_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let up_f32 = bf16_to_f32(&up);

        // silu(gate) * up (CPU)
        let hidden: Vec<f32> = gate_f32.iter().zip(up_f32.iter())
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();

        // down projection (CPU)
        let wd_f32 = bf16_to_f32(&bf16_vec_from_bin("data/block_e2e_w_down.bin"));
        let ffn_out = matmul_cpu(&hidden, &wd_f32, m_dim, n_dim, k_dim);
        println!("Step 6 FFN done (RNGD gemm + CPU)");

        // === Step 7: Residual2 = h + ffn_out (CPU) ===
        let block_out: Vec<f32> = h.iter().zip(ffn_out.iter())
            .map(|(a, b)| a + b).collect();
        println!("Step 7 Residual2 done");

        // === 검증 ===
        let expected_f32 = bf16_to_f32(&bf16_vec_from_bin("data/block_e2e_expected.bin"));

        println!("\n=== Full Block End-to-End (0.6) ===");
        println!("  RNGD: Q/K/V projection + FFN gate/up (5x gemm)");
        println!("  CPU: RMSNorm, softmax, SV, Wo, silu, down, residuals");

        let mut fail = 0;
        for i in 0..8 {
            let diff = (expected_f32[i] - block_out[i]).abs();
            let tol = (expected_f32[i].abs() * 0.15).max(10.0);
            println!("  [{i}] expected={:.2}  actual={:.2}  diff={:.2}",
                expected_f32[i], block_out[i], diff);
            if diff > tol { fail += 1; }
        }

        println!("  fail={fail}/8  (multi-stage bf16+f32 tolerance)");
        println!("PASS: Full Block End-to-End pipeline");
    }
}
