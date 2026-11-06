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
        let ms: f32 = row.iter().map(|v| v*v).sum::<f32>() / dim as f32;
        let rms = (ms + eps).sqrt();
        for c in 0..dim { out[r*dim+c] = row[c] / rms * weight[c]; }
    }
    out
}
fn silu_cpu(x: &[f32]) -> Vec<f32> { x.iter().map(|v| v / (1.0 + (-v).exp())).collect() }
fn elemwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x*y).collect() }
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x+y).collect() }

#[tokio::main]
async fn main() { println!("Block dim128 end-to-end test"); }

#[cfg(test)]
mod tests {
    use super::*;

    async fn npu_gemm(ctx: &mut Context, x_buf: Vec<u8>, w_buf: Vec<u8>) -> Vec<bf16> {
        let x_hbm = HostTensor::<bf16, m![M, K]>::from_buf(x_buf).to_hbm(&mut ctx.pdma).await;
        let w_hbm = HostTensor::<bf16, m![N, K]>::from_buf(w_buf).to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_ffn_gate_kernel, (ctx, &x_hbm, &w_hbm)).await;
        out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec()
    }

    /// K-tiled GEMM: x[seq, k_total] × w[n_out, k_total].T
    /// x_tiles: 각 타일의 .bin 경로, w_tiles: 각 타일의 .bin 경로
    async fn k_tiled_gemm(ctx: &mut Context, x_tiles: &[Vec<u8>], w_tiles: &[Vec<u8>],
                          seq: usize, n_out: usize) -> Vec<f32> {
        let pad_n: usize = 512;
        let n_tiles = x_tiles.len();
        let mut accum = vec![0.0f32; seq * n_out];
        for t in 0..n_tiles {
            let raw = npu_gemm(ctx, x_tiles[t].clone(), w_tiles[t].clone()).await;
            let padded = bf16_to_f32(&raw);
            for s in 0..seq {
                for d in 0..n_out {
                    accum[s * n_out + d] += padded[s * pad_n + d];
                }
            }
        }
        accum
    }

    /// 입력 데이터를 K축 타일로 분할하여 [512,64] bf16 패킹
    fn tile_input(data: &[f32], seq: usize, full_cols: usize, tile_idx: usize, tile_k: usize) -> Vec<u8> {
        let start = tile_idx * tile_k;
        let mut tiled = vec![0.0f32; seq * tile_k];
        for s in 0..seq {
            for c in 0..tile_k {
                if start + c < full_cols {
                    tiled[s * tile_k + c] = data[s * full_cols + start + c];
                }
            }
        }
        pack_bf16(&tiled, seq, tile_k, 512, 64)
    }

    #[tokio::test]
    async fn block_dim128_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();

        let dim: usize = 128;
        let hidden: usize = 256;
        let n_heads: usize = 8;
        let n_kv_heads: usize = 2;
        let head_dim: usize = 16;
        let seq: usize = 8;
        let n_rep: usize = n_heads / n_kv_heads;
        let eps: f32 = 1e-5;
        let kv_dim: usize = n_kv_heads * head_dim; // 32

        // Load input [8, 128] from 2 tiles
        let x_t0_f32 = extract_region(&bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_x_t0.bin")), 64, seq, 64);
        let x_t1_f32 = extract_region(&bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_x_t1.bin")), 64, seq, 64);
        let mut x_f32 = vec![0.0f32; seq * dim];
        for s in 0..seq {
            for c in 0..64 { x_f32[s*dim + c] = x_t0_f32[s*64 + c]; }
            for c in 0..64 { x_f32[s*dim + 64 + c] = x_t1_f32[s*64 + c]; }
        }
        println!("Input x[0:4]: {:.4?}", &x_f32[..4]);

        // Load norm weights
        let norm1_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_input_norm.bin"));
        let norm2_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_post_attn_norm.bin"));

        // ── Step 1: RMSNorm #1 (CPU) ──
        let normed = rmsnorm_cpu(&x_f32, &norm1_w, dim, eps);
        println!("Step 1: RMSNorm #1. normed[0:4]={:.4?}", &normed[..4]);

        // ── Step 2: Q/K/V projection (NPU, K-tiling 2타일 each) ──
        let n_t0 = tile_input(&normed, seq, dim, 0, 64);
        let n_t1 = tile_input(&normed, seq, dim, 1, 64);

        let q = k_tiled_gemm(&mut ctx,
            &[n_t0.clone(), n_t1.clone()],
            &[load_bf16_bin("data/block_dim128_q_proj_t0.bin"),
              load_bf16_bin("data/block_dim128_q_proj_t1.bin")],
            seq, dim).await;
        println!("Step 2a: Q proj (NPU 2-tile). Q[0:4]={:.4?}", &q[..4]);

        let k = k_tiled_gemm(&mut ctx,
            &[n_t0.clone(), n_t1.clone()],
            &[load_bf16_bin("data/block_dim128_k_proj_t0.bin"),
              load_bf16_bin("data/block_dim128_k_proj_t1.bin")],
            seq, kv_dim).await;
        println!("Step 2b: K proj (NPU 2-tile). K[0:4]={:.4?}", &k[..4]);

        let v = k_tiled_gemm(&mut ctx,
            &[n_t0.clone(), n_t1.clone()],
            &[load_bf16_bin("data/block_dim128_v_proj_t0.bin"),
              load_bf16_bin("data/block_dim128_v_proj_t1.bin")],
            seq, kv_dim).await;
        println!("Step 2c: V proj (NPU 2-tile). V[0:4]={:.4?}", &v[..4]);

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
                        dot += q[i*dim + h*head_dim + d] * k[j*kv_dim + kv_h*head_dim + d];
                    }
                    scores[i*seq + j] = dot * scale;
                }
            }
            for i in 0..seq {
                let row = &mut scores[i*seq..(i+1)*seq];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sm = 0.0f32;
                for v in row.iter_mut() { *v = (*v - mx).exp(); sm += *v; }
                for v in row.iter_mut() { *v /= sm; }
            }
            for i in 0..seq {
                for d in 0..head_dim {
                    let mut sm = 0.0f32;
                    for j in 0..seq { sm += scores[i*seq+j] * v[j*kv_dim + kv_h*head_dim + d]; }
                    attn_out[i*dim + h*head_dim + d] = sm;
                }
            }
        }
        println!("Step 3: GQA Attention (CPU). attn_out[0:4]={:.4?}", &attn_out[..4]);

        // ── Step 4: O projection (NPU, K-tiling 2타일) ──
        let ao_t0 = tile_input(&attn_out, seq, dim, 0, 64);
        let ao_t1 = tile_input(&attn_out, seq, dim, 1, 64);
        let attn_proj = k_tiled_gemm(&mut ctx,
            &[ao_t0, ao_t1],
            &[load_bf16_bin("data/block_dim128_o_proj_t0.bin"),
              load_bf16_bin("data/block_dim128_o_proj_t1.bin")],
            seq, dim).await;
        println!("Step 4: O proj (NPU 2-tile). attn_proj[0:4]={:.4?}", &attn_proj[..4]);

        // ── Step 5: Residual #1 ──
        let h1 = vec_add(&x_f32, &attn_proj);
        println!("Step 5: Residual #1. h1[0:4]={:.4?}", &h1[..4]);

        // ── Step 6: RMSNorm #2 ──
        let normed2 = rmsnorm_cpu(&h1, &norm2_w, dim, eps);
        println!("Step 6: RMSNorm #2. normed2[0:4]={:.4?}", &normed2[..4]);

        // ── Step 7: FFN Gate + Up (NPU, K-tiling 2타일 each) ──
        let n2_t0 = tile_input(&normed2, seq, dim, 0, 64);
        let n2_t1 = tile_input(&normed2, seq, dim, 1, 64);

        let gate = k_tiled_gemm(&mut ctx,
            &[n2_t0.clone(), n2_t1.clone()],
            &[load_bf16_bin("data/block_dim128_ffn_gate_t0.bin"),
              load_bf16_bin("data/block_dim128_ffn_gate_t1.bin")],
            seq, hidden).await;
        println!("Step 7a: FFN gate (NPU 2-tile). gate[0:4]={:.4?}", &gate[..4]);

        let up = k_tiled_gemm(&mut ctx,
            &[n2_t0, n2_t1],
            &[load_bf16_bin("data/block_dim128_ffn_up_t0.bin"),
              load_bf16_bin("data/block_dim128_ffn_up_t1.bin")],
            seq, hidden).await;
        println!("Step 7b: FFN up (NPU 2-tile). up[0:4]={:.4?}", &up[..4]);

        // ── Step 8: SiLU + Gate×Up (CPU) ──
        let activated = silu_cpu(&gate);
        let ffn_mid = elemwise_mul(&activated, &up);
        println!("Step 8: SiLU + mul (CPU). ffn_mid[0:4]={:.4?}", &ffn_mid[..4]);

        // ── Step 9: FFN Down (NPU, K-tiling 4타일) ──
        let m_t0 = tile_input(&ffn_mid, seq, hidden, 0, 64);
        let m_t1 = tile_input(&ffn_mid, seq, hidden, 1, 64);
        let m_t2 = tile_input(&ffn_mid, seq, hidden, 2, 64);
        let m_t3 = tile_input(&ffn_mid, seq, hidden, 3, 64);

        let down = k_tiled_gemm(&mut ctx,
            &[m_t0, m_t1, m_t2, m_t3],
            &[load_bf16_bin("data/block_dim128_ffn_down_t0.bin"),
              load_bf16_bin("data/block_dim128_ffn_down_t1.bin"),
              load_bf16_bin("data/block_dim128_ffn_down_t2.bin"),
              load_bf16_bin("data/block_dim128_ffn_down_t3.bin")],
            seq, dim).await;
        println!("Step 9: FFN down (NPU 4-tile). down[0:4]={:.4?}", &down[..4]);

        // ── Step 10: Residual #2 ──
        let output = vec_add(&h1, &down);
        println!("Step 10: Residual #2. output[0:4]={:.4?}", &output[..4]);

        // ── 검증 ──
        let exp_t0 = extract_region(&bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_expected_t0.bin")), 64, seq, 64);
        let exp_t1 = extract_region(&bf16_to_f32(&bf16_vec_from_bin("data/block_dim128_expected_t1.bin")), 64, seq, 64);
        let mut expected = vec![0.0f32; seq * dim];
        for s in 0..seq {
            for c in 0..64 { expected[s*dim + c] = exp_t0[s*64 + c]; }
            for c in 0..64 { expected[s*dim + 64 + c] = exp_t1[s*64 + c]; }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let mut max_diff: f32 = 0.0;
        let mut ref_max: f32 = 0.0;
        for i in 0..seq*dim {
            let diff = (expected[i] - output[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if expected[i].abs() > ref_max { ref_max = expected[i].abs(); }
        }
        let rel_err = if ref_max > 0.0 { max_diff / ref_max * 100.0 } else { 0.0 };

        println!("\n=== Block dim128 Full E2E ===");
        println!("  dim={dim}, hidden={hidden}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}");
        println!("  NPU GEMM: Q/K/V(2t×3) + O(2t) + gate/up(2t×2) + down(4t) = 18회");
        println!("  elapsed: {:.2}s", elapsed);
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);

        for i in 0..8.min(seq*dim) {
            println!("  [{}] expected={:.6}  actual={:.6}  diff={:.6}",
                i, expected[i], output[i], (expected[i]-output[i]).abs());
        }

        let tol = (ref_max * 0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass { "PASS" } else { "FAIL" });
        assert!(pass, "block_dim128 FAIL: max_diff={max_diff} > tol={tol}");
    }
}
