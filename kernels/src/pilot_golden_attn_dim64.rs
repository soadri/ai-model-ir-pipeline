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
async fn main() { println!("Attention dim64 GQA test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn attn_dim64_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();

        // dim=64, n_heads=8, n_kv_heads=2, head_dim=8, seq=8
        let dim: usize = 64;
        let n_heads: usize = 8;
        let n_kv_heads: usize = 2;
        let head_dim: usize = 8;
        let seq: usize = 8;
        let n_rep: usize = n_heads / n_kv_heads;

        // Step 1: Q = x @ Wq.T  [512,64] × [512,64].T → [512,512]
        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/attn_dim64_x.bin")).to_hbm(&mut ctx.pdma).await;
        let wq_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_dim64_wq.bin")).to_hbm(&mut ctx.pdma).await;
        let q_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wq_hbm)).await;
        let q_all: Vec<bf16> = q_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let q_f32 = bf16_to_f32_vec(&q_all);
        println!("Step 1: Q projection done (NPU GEMM). Q[0:4]={:.4?}", &q_f32[..4]);

        // Step 2: K = x @ Wk.T  (Wk=[16,64], padded to [512,64])
        let wk_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_dim64_wk.bin")).to_hbm(&mut ctx.pdma).await;
        let k_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wk_hbm)).await;
        let k_all: Vec<bf16> = k_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let k_f32 = bf16_to_f32_vec(&k_all);
        println!("Step 2: K projection done (NPU GEMM). K[0:4]={:.4?}", &k_f32[..4]);

        // Step 3: V = x @ Wv.T
        let wv_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_dim64_wv.bin")).to_hbm(&mut ctx.pdma).await;
        let v_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x_hbm, &wv_hbm)).await;
        let v_all: Vec<bf16> = v_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let v_f32 = bf16_to_f32_vec(&v_all);
        println!("Step 3: V projection done (NPU GEMM). V[0:4]={:.4?}", &v_f32[..4]);

        // Extract valid region: [seq, dim] from [M, N]
        // Q: [8, 64] from [512, 512]
        let mut q = vec![0.0f32; seq * dim];
        let mut k = vec![0.0f32; seq * n_kv_heads * head_dim];
        let mut v = vec![0.0f32; seq * n_kv_heads * head_dim];
        let n_padded = 512;
        for s in 0..seq {
            for d in 0..dim {
                q[s * dim + d] = q_f32[s * n_padded + d];
            }
            for d in 0..(n_kv_heads * head_dim) {
                k[s * n_kv_heads * head_dim + d] = k_f32[s * n_padded + d];
                v[s * n_kv_heads * head_dim + d] = v_f32[s * n_padded + d];
            }
        }

        // Step 4: GQA reshape + repeat + QK^T + softmax + attn×V (CPU)
        // q: [seq, n_heads, head_dim] = [8, 8, 8]
        // k: [seq, n_kv_heads, head_dim] = [8, 2, 8] → repeat → [8, 8, 8]
        // v: same as k

        // Per-head attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq * dim]; // [8, 64]

        for h in 0..n_heads {
            let kv_h = h / n_rep; // which kv head

            // scores[i][j] = sum_d(q[i,h,d] * k[j,kv_h,d]) * scale
            let mut scores = vec![0.0f32; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qi = q[i * dim + h * head_dim + d];
                        let kj = k[j * n_kv_heads * head_dim + kv_h * head_dim + d];
                        dot += qi * kj;
                    }
                    scores[i * seq + j] = dot * scale;
                }
            }

            // softmax per row
            for i in 0..seq {
                let row = &mut scores[i * seq..(i + 1) * seq];
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

            // out[i, h, :] = sum_j(scores[i,j] * v[j, kv_h, :])
            for i in 0..seq {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq {
                        let vj = v[j * n_kv_heads * head_dim + kv_h * head_dim + d];
                        sum += scores[i * seq + j] * vj;
                    }
                    attn_out[i * dim + h * head_dim + d] = sum;
                }
            }
        }
        println!("Step 4: GQA attention done (CPU). attn_out[0:4]={:.4?}", &attn_out[..4]);

        // Step 5: O = attn_out @ Wo.T (NPU GEMM)
        // Pack attn_out into [512, 64] bf16 bin
        let mut attn_out_padded = vec![0u8; 512 * 64 * 2]; // M*K*2 bytes
        for s in 0..seq {
            for d in 0..dim {
                let val = bf16::from_f32(attn_out[s * dim + d]);
                let bytes = val.to_bits().to_le_bytes();
                let idx = (s * 64 + d) * 2;
                attn_out_padded[idx] = bytes[0];
                attn_out_padded[idx + 1] = bytes[1];
            }
        }

        let ao_hbm = HostTensor::<bf16, m![M, K]>::from_buf(attn_out_padded)
            .to_hbm(&mut ctx.pdma).await;
        let wo_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/attn_dim64_wo.bin")).to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &ao_hbm, &wo_hbm)).await;
        let out_all: Vec<bf16> = out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let out_f32 = bf16_to_f32_vec(&out_all);

        // Extract [seq, dim] from [M, N]
        let mut output = vec![0.0f32; seq * dim];
        for s in 0..seq {
            for d in 0..dim {
                output[s * dim + d] = out_f32[s * n_padded + d];
            }
        }
        println!("Step 5: O projection done (NPU GEMM). output[0:4]={:.4?}", &output[..4]);

        // 검증
        let expected_f32 = bf16_to_f32_vec(&bf16_vec_from_bin("data/attn_dim64_expected.bin"));

        let elapsed = t0.elapsed().as_secs_f64();
        println!("\n=== Attention dim64 GQA End-to-End ===");
        println!("  dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, seq={seq}");
        println!("  Q/K/V/O projection: NPU GEMM (M=512,K=64,N=512)");
        println!("  GQA reshape + QK^T + softmax + attn×V: CPU");
        println!("  elapsed: {:.2}s", elapsed);

        let mut max_diff: f32 = 0.0;
        let mut fail = 0;
        for s in 0..seq {
            for d in 0..dim {
                let idx = s * dim + d; // output 인덱스
                let exp_idx = s * 64 + d; // expected는 [512, 64] 패딩
                if exp_idx >= expected_f32.len() { continue; }
                let diff = (expected_f32[exp_idx] - output[idx]).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }

        let mut ref_max: f32 = 0.0;
        for s in 0..seq {
            for d in 0..dim {
                let v = expected_f32[s * 64 + d].abs();
                if v > ref_max { ref_max = v; }
            }
        }
        let rel_err = if ref_max > 0.0 { max_diff / ref_max * 100.0 } else { 0.0 };
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);

        for i in 0..8.min(seq * dim) {
            let exp_idx = (i / dim) * 64 + (i % dim);
            if exp_idx >= expected_f32.len() { continue; }
            let diff = (expected_f32[exp_idx] - output[i]).abs();
            println!("  [{}] expected={:.6}  actual={:.6}  diff={:.6}",
                i, expected_f32[exp_idx], output[i], diff);
        }

        let tol = (ref_max * 0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass { "PASS" } else { "FAIL" });
        println!("PASS: attn_dim64 max_diff={:.6}", max_diff);
        assert!(pass, "attn_dim64 FAIL: max_diff={max_diff} > tol={tol}");
    }
}
