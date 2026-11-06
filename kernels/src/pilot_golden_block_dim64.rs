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

fn pack_bf16(data: &[f32], rows: usize, cols: usize, pad_rows: usize, pad_cols: usize) -> Vec<u8> {
    let mut buf = vec![0u8; pad_rows * pad_cols * 2];
    for r in 0..rows.min(pad_rows) {
        for c in 0..cols.min(pad_cols) {
            let val = bf16::from_f32(data[r * cols + c]);
            let bytes = val.to_bits().to_le_bytes();
            let idx = (r * pad_cols + c) * 2;
            buf[idx] = bytes[0];
            buf[idx + 1] = bytes[1];
        }
    }
    buf
}

fn extract_region(padded: &[f32], pad_cols: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = padded[r * pad_cols + c];
        }
    }
    out
}

fn rmsnorm_cpu(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r*dim..(r+1)*dim];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (mean_sq + eps).sqrt();
        for c in 0..dim {
            out[r*dim + c] = row[c] / rms * weight[c];
        }
    }
    out
}

fn silu_cpu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

fn elemwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[tokio::main]
async fn main() { println!("Block dim64 end-to-end test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    async fn npu_gemm(ctx: &mut Context, x_buf: Vec<u8>, w_buf: Vec<u8>) -> Vec<bf16> {
        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(x_buf)
            .to_hbm(&mut ctx.pdma).await;
        let w_hbm = HostTensor::<bf16, m![N, K]>::from_buf(w_buf)
            .to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_ffn_gate_kernel, (ctx, &x_hbm, &w_hbm)).await;
        out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec()
    }

    #[tokio::test]
    async fn block_dim64_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();

        let dim: usize = 64;
        let hidden: usize = 128;
        let n_heads: usize = 8;
        let n_kv_heads: usize = 2;
        let head_dim: usize = 8;
        let seq: usize = 8;
        let n_rep: usize = n_heads / n_kv_heads;
        let eps: f32 = 1e-5;
        let pad_n: usize = 512;

        // Load input
        let x_f32 = extract_region(
            &bf16_to_f32(&bf16_vec_from_bin("data/block_dim64_x.bin")), 64, seq, dim);
        println!("Input x[0:4]: {:.4?}", &x_f32[..4]);

        // Load norm weights
        let norm1_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim64_input_norm.bin"));
        let norm2_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim64_post_attention_norm.bin"));

        // ── Step 1: RMSNorm #1 (CPU) ──
        let normed = rmsnorm_cpu(&x_f32, &norm1_w, dim, eps);
        println!("Step 1: RMSNorm #1 done. normed[0:4]={:.4?}", &normed[..4]);

        // ── Step 2: Q/K/V projection (NPU GEMM × 3) ──
        let normed_buf = pack_bf16(&normed, seq, dim, 512, 64);

        let q_raw = npu_gemm(&mut ctx, normed_buf.clone(),
            load_bf16_bin("data/block_dim64_q_proj.bin")).await;
        let q_padded = bf16_to_f32(&q_raw);
        let q = extract_region(&q_padded, pad_n, seq, dim);
        println!("Step 2a: Q proj (NPU). Q[0:4]={:.4?}", &q[..4]);

        let k_raw = npu_gemm(&mut ctx, normed_buf.clone(),
            load_bf16_bin("data/block_dim64_k_proj.bin")).await;
        let k_padded = bf16_to_f32(&k_raw);
        let k = extract_region(&k_padded, pad_n, seq, n_kv_heads * head_dim);
        println!("Step 2b: K proj (NPU). K[0:4]={:.4?}", &k[..4]);

        let v_raw = npu_gemm(&mut ctx, normed_buf,
            load_bf16_bin("data/block_dim64_v_proj.bin")).await;
        let v_padded = bf16_to_f32(&v_raw);
        let v = extract_region(&v_padded, pad_n, seq, n_kv_heads * head_dim);
        println!("Step 2c: V proj (NPU). V[0:4]={:.4?}", &v[..4]);

        // ── Step 3: GQA Attention (CPU) ──
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq * dim];

        for h in 0..n_heads {
            let kv_h = h / n_rep;
            let mut scores = vec![0.0f32; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i*dim + h*head_dim + d] * k[j*n_kv_heads*head_dim + kv_h*head_dim + d];
                    }
                    scores[i*seq + j] = dot * scale;
                }
            }
            // softmax
            for i in 0..seq {
                let row = &mut scores[i*seq..(i+1)*seq];
                let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() { *val = (*val - max_v).exp(); sum += *val; }
                for val in row.iter_mut() { *val /= sum; }
            }
            for i in 0..seq {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq {
                        sum += scores[i*seq+j] * v[j*n_kv_heads*head_dim + kv_h*head_dim + d];
                    }
                    attn_out[i*dim + h*head_dim + d] = sum;
                }
            }
        }
        println!("Step 3: GQA Attention (CPU). attn_out[0:4]={:.4?}", &attn_out[..4]);

        // ── Step 4: O projection (NPU GEMM) ──
        let ao_buf = pack_bf16(&attn_out, seq, dim, 512, 64);
        let o_raw = npu_gemm(&mut ctx, ao_buf,
            load_bf16_bin("data/block_dim64_o_proj.bin")).await;
        let o_padded = bf16_to_f32(&o_raw);
        let attn_proj = extract_region(&o_padded, pad_n, seq, dim);
        println!("Step 4: O proj (NPU). attn_proj[0:4]={:.4?}", &attn_proj[..4]);

        // ── Step 5: Residual Add #1 (CPU) ──
        let h1 = vec_add(&x_f32, &attn_proj);
        println!("Step 5: Residual #1. h1[0:4]={:.4?}", &h1[..4]);

        // ── Step 6: RMSNorm #2 (CPU) ──
        let normed2 = rmsnorm_cpu(&h1, &norm2_w, dim, eps);
        println!("Step 6: RMSNorm #2. normed2[0:4]={:.4?}", &normed2[..4]);

        // ── Step 7: FFN Gate + Up (NPU GEMM × 2) ──
        let normed2_buf = pack_bf16(&normed2, seq, dim, 512, 64);

        let gate_raw = npu_gemm(&mut ctx, normed2_buf.clone(),
            load_bf16_bin("data/block_dim64_ffn_gate.bin")).await;
        let gate_padded = bf16_to_f32(&gate_raw);
        let gate = extract_region(&gate_padded, pad_n, seq, hidden);
        println!("Step 7a: FFN gate (NPU). gate[0:4]={:.4?}", &gate[..4]);

        let up_raw = npu_gemm(&mut ctx, normed2_buf,
            load_bf16_bin("data/block_dim64_ffn_up.bin")).await;
        let up_padded = bf16_to_f32(&up_raw);
        let up = extract_region(&up_padded, pad_n, seq, hidden);
        println!("Step 7b: FFN up (NPU). up[0:4]={:.4?}", &up[..4]);

        // ── Step 8: SiLU + Gate×Up (CPU) ──
        let activated = silu_cpu(&gate);
        let ffn_mid = elemwise_mul(&activated, &up);
        println!("Step 8: SiLU + mul (CPU). ffn_mid[0:4]={:.4?}", &ffn_mid[..4]);

        // ── Step 9: FFN Down (NPU GEMM × 2, K-tiling) ──
        // ffn_mid: [seq, 128] → 2 tiles of [seq, 64]
        let mid_t0 = pack_bf16(
            &ffn_mid.iter().enumerate()
                .filter(|(i, _)| (*i % hidden) < 64)
                .map(|(_, v)| *v).collect::<Vec<f32>>(),
            seq, 64, 512, 64);
        let mid_t1 = pack_bf16(
            &ffn_mid.iter().enumerate()
                .filter(|(i, _)| (*i % hidden) >= 64)
                .map(|(_, v)| *v).collect::<Vec<f32>>(),
            seq, 64, 512, 64);

        let d0_raw = npu_gemm(&mut ctx, mid_t0,
            load_bf16_bin("data/block_dim64_ffn_down_w0.bin")).await;
        let d0 = bf16_to_f32(&d0_raw);

        let d1_raw = npu_gemm(&mut ctx, mid_t1,
            load_bf16_bin("data/block_dim64_ffn_down_w1.bin")).await;
        let d1 = bf16_to_f32(&d1_raw);

        // Accumulate tiles
        let mut down = vec![0.0f32; seq * dim];
        for s in 0..seq {
            for d in 0..dim {
                down[s*dim + d] = d0[s*pad_n + d] + d1[s*pad_n + d];
            }
        }
        println!("Step 9: FFN down (NPU, 2-tile). down[0:4]={:.4?}", &down[..4]);

        // ── Step 10: Residual Add #2 (CPU) ──
        let output = vec_add(&h1, &down);
        println!("Step 10: Residual #2. output[0:4]={:.4?}", &output[..4]);

        // ── 검증 ──
        let expected_all = bf16_to_f32(&bf16_vec_from_bin("data/block_dim64_expected.bin"));

        let elapsed = t0.elapsed().as_secs_f64();
        println!("\n=== Block dim64 Full E2E ===");
        println!("  dim={dim}, hidden={hidden}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");
        println!("  NPU GEMM: Q/K/V/O + gate/up + down(2-tile) = 9회");
        println!("  CPU: RMSNorm×2, GQA attention, SiLU, residual×2");
        println!("  elapsed: {:.2}s", elapsed);

        let mut max_diff: f32 = 0.0;
        for s in 0..seq {
            for d in 0..dim {
                let exp = expected_all[s * 64 + d];
                let act = output[s * dim + d];
                let diff = (exp - act).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }

        let mut ref_max: f32 = 0.0;
        for s in 0..seq {
            for d in 0..dim {
                let v = expected_all[s * 64 + d].abs();
                if v > ref_max { ref_max = v; }
            }
        }
        let rel_err = if ref_max > 0.0 { max_diff / ref_max * 100.0 } else { 0.0 };
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);

        for i in 0..8 {
            let s = i / dim;
            let d = i % dim;
            let exp = expected_all[s * 64 + d];
            let act = output[i];
            println!("  [{}] expected={:.6}  actual={:.6}  diff={:.6}", i, exp, act, (exp-act).abs());
        }

        let tol = (ref_max * 0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass { "PASS" } else { "FAIL" });
        println!("PASS: block_dim64 max_diff={:.6}", max_diff);
        assert!(pass, "block_dim64 FAIL: max_diff={max_diff} > tol={tol}");
    }
}
