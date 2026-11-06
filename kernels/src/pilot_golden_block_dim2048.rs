use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_ffn_gate_kernel::{
    M, K, N, pilot_golden_ffn_gate_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}
fn bf16_vec_from_bin(path: &str) -> Vec<bf16> {
    load_bf16_bin(path).chunks(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn bf16_to_f32(data: &[bf16]) -> Vec<f32> { data.iter().map(|v| f32::from(*v)).collect() }
fn pack_bf16(data: &[f32], rows: usize, cols: usize, pr: usize, pc: usize) -> Vec<u8> {
    let mut buf = vec![0u8; pr*pc*2];
    for r in 0..rows.min(pr) { for c in 0..cols.min(pc) {
        let v = bf16::from_f32(data[r*cols+c]); let b = v.to_bits().to_le_bytes();
        let i = (r*pc+c)*2; buf[i]=b[0]; buf[i+1]=b[1];
    }} buf
}
fn extract_region(p: &[f32], pc: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; rows*cols];
    for r in 0..rows { for c in 0..cols { o[r*cols+c] = p[r*pc+c]; } } o
}
fn rmsnorm_cpu(x: &[f32], w: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = x.len()/dim; let mut o = vec![0.0f32; x.len()];
    for r in 0..rows { let row = &x[r*dim..(r+1)*dim];
        let ms: f32 = row.iter().map(|v| v*v).sum::<f32>() / dim as f32;
        let rms = (ms+eps).sqrt();
        for c in 0..dim { o[r*dim+c] = row[c]/rms*w[c]; }
    } o
}
fn silu_cpu(x: &[f32]) -> Vec<f32> { x.iter().map(|v| v/(1.0+(-v).exp())).collect() }
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x*y).collect() }
fn vadd(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x+y).collect() }

#[tokio::main]
async fn main() { println!("Block dim2048 E2E (optimized)"); }

#[cfg(test)]
mod tests {
    use super::*;
    const PN: usize = 512;

    async fn npu_gemm_from_hbm(ctx: &mut Context,
        x_hbm: &HbmTensor<bf16, m![1], m![M, K]>,
        w_buf: Vec<u8>) -> Vec<bf16> {
        let wh = HostTensor::<bf16, m![N,K]>::from_buf(w_buf).to_hbm(&mut ctx.pdma).await;
        let oh = launch(pilot_golden_ffn_gate_kernel, (ctx, x_hbm, &wh)).await;
        oh.to_host::<m![M,N]>(&mut ctx.pdma).await.into_vec()
    }

    /// 최적화: 입력 HBM 텐서를 K-타일마다 1번만 업로드
    async fn k_tiled_gemm_opt(ctx: &mut Context, x_bufs: &[Vec<u8>], w_bufs: &[Vec<u8>],
                              seq: usize, nout: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; seq*nout];
        for t in 0..x_bufs.len() {
            let xh = HostTensor::<bf16, m![M,K]>::from_buf(x_bufs[t].clone())
                .to_hbm(&mut ctx.pdma).await;
            let raw = npu_gemm_from_hbm(ctx, &xh, w_bufs[t].clone()).await;
            let p = bf16_to_f32(&raw);
            for s in 0..seq { for d in 0..nout { acc[s*nout+d] += p[s*PN+d]; } }
        }
        acc
    }

    /// 최적화: K+N 타일링에서 입력 HBM을 N-타일 간 재사용
    async fn kn_tiled_gemm_opt(ctx: &mut Context, input: &[f32], seq: usize, in_dim: usize,
                               prefix: &str, n_out: usize, kt: usize, n_tiles: usize) -> Vec<f32> {
        // Step 1: 입력 K-타일을 HBM에 1번만 업로드
        let x_hbms: Vec<_> = (0..kt).map(|t| {
            let st = t*64;
            let mut tiled = vec![0.0f32; seq*64];
            for s in 0..seq { for c in 0..64 {
                if st+c < in_dim { tiled[s*64+c] = input[s*in_dim+st+c]; }
            }}
            pack_bf16(&tiled, seq, 64, 512, 64)
        }).collect();

        // 입력을 HBM에 업로드 (kt번만, N-타일 수와 무관)
        let mut x_hbm_tensors = Vec::with_capacity(kt);
        for buf in &x_hbms {
            let hbm = HostTensor::<bf16, m![M,K]>::from_buf(buf.clone())
                .to_hbm(&mut ctx.pdma).await;
            x_hbm_tensors.push(hbm);
        }

        // Step 2: N-타일 순회 — 입력 HBM 재사용!
        let mut result = vec![0.0f32; seq * n_out];
        for nt in 0..n_tiles {
            let n_start = nt * 512;
            let n_cols = 512.min(n_out - n_start);

            // K-타일 순회: 가중치만 새로 업로드
            let mut acc = vec![0.0f32; seq * n_cols];
            for t in 0..kt {
                let w_buf = load_bf16_bin(&format!("data/{prefix}_n{nt}_t{t}.bin"));
                let raw = npu_gemm_from_hbm(ctx, &x_hbm_tensors[t], w_buf).await;
                let p = bf16_to_f32(&raw);
                for s in 0..seq { for d in 0..n_cols { acc[s*n_cols+d] += p[s*PN+d]; } }
            }

            for s in 0..seq { for d in 0..n_cols {
                result[s*n_out + n_start + d] = acc[s*n_cols + d];
            }}
        }
        result
    }

    /// K-only (N≤512): prefix_t{kt}
    async fn k_only_tiled_opt(ctx: &mut Context, input: &[f32], seq: usize, in_dim: usize,
                              prefix: &str, n_out: usize, kt: usize) -> Vec<f32> {
        let x_bufs: Vec<Vec<u8>> = (0..kt).map(|t| {
            let st = t*64;
            let mut tiled = vec![0.0f32; seq*64];
            for s in 0..seq { for c in 0..64 {
                if st+c < in_dim { tiled[s*64+c] = input[s*in_dim+st+c]; }
            }}
            pack_bf16(&tiled, seq, 64, 512, 64)
        }).collect();
        let w_bufs: Vec<Vec<u8>> = (0..kt).map(|t|
            load_bf16_bin(&format!("data/{prefix}_t{t}.bin"))).collect();
        k_tiled_gemm_opt(ctx, &x_bufs, &w_bufs, seq, n_out).await
    }

    fn load_tiles_multi(prefix: &str, seq: usize, dim: usize, kt: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq*dim];
        for t in 0..kt {
            let tile = extract_region(&bf16_to_f32(&bf16_vec_from_bin(
                &format!("data/{prefix}_t{t}.bin"))), 64, seq, 64);
            for s in 0..seq { for c in 0..64 { out[s*dim+t*64+c] = tile[s*64+c]; } }
        }
        out
    }

    fn load_norm_weight(prefix: &str, dim: usize) -> Vec<f32> {
        let np = (dim+511)/512; let mut w = vec![0.0f32; dim];
        for p in 0..np {
            let part = bf16_to_f32(&bf16_vec_from_bin(&format!("data/{prefix}_p{p}.bin")));
            let s = p*512; let c = 512.min(dim-s);
            for i in 0..c { w[s+i] = part[i]; }
        }
        w
    }

    #[tokio::test]
    async fn block_dim2048_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();
        let dim=2048usize; let hidden=4096usize; let nh=8usize; let nkv=2usize;
        let hd=256usize; let seq=8usize; let nrep=nh/nkv; let eps=1e-5f32;
        let kvd=nkv*hd; let kt=dim/64;
        let q_nt=(dim+511)/512; let gate_nt=(hidden+511)/512;
        let down_kt=hidden/64; let down_nt=(dim+511)/512;

        let x_f32 = load_tiles_multi("block_dim2048_x", seq, dim, kt);
        println!("Input loaded. x[0:4]: {:.4?}", &x_f32[..4]);

        let norm1_w = load_norm_weight("block_dim2048_input_norm", dim);
        let norm2_w = load_norm_weight("block_dim2048_post_attn_norm", dim);

        // Step 1: RMSNorm #1
        let normed = rmsnorm_cpu(&x_f32, &norm1_w, dim, eps);
        println!("Step 1: RMSNorm #1 [{:.1}s]", t0.elapsed().as_secs_f64());

        // Step 2: Q/K/V projection
        let q = kn_tiled_gemm_opt(&mut ctx, &normed, seq, dim,
            "block_dim2048_q_proj", dim, kt, q_nt).await;
        println!("Step 2a: Q proj ({kt}Kt×{q_nt}Nt) [{:.1}s]", t0.elapsed().as_secs_f64());

        let k = k_only_tiled_opt(&mut ctx, &normed, seq, dim,
            "block_dim2048_k_proj", kvd, kt).await;
        println!("Step 2b: K proj ({kt}Kt) [{:.1}s]", t0.elapsed().as_secs_f64());

        let v = k_only_tiled_opt(&mut ctx, &normed, seq, dim,
            "block_dim2048_v_proj", kvd, kt).await;
        println!("Step 2c: V proj ({kt}Kt) [{:.1}s]", t0.elapsed().as_secs_f64());

        // Step 3: GQA Attention (CPU)
        let scale = 1.0/(hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq*dim];
        for h in 0..nh {
            let kvh = h/nrep;
            let mut sc = vec![0.0f32; seq*seq];
            for i in 0..seq { for j in 0..seq {
                let mut dot = 0.0f32;
                for d in 0..hd { dot += q[i*dim+h*hd+d]*k[j*kvd+kvh*hd+d]; }
                sc[i*seq+j] = dot*scale;
            }}
            for i in 0..seq {
                let row = &mut sc[i*seq..(i+1)*seq];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sm = 0.0f32;
                for vl in row.iter_mut() { *vl = (*vl-mx).exp(); sm += *vl; }
                for vl in row.iter_mut() { *vl /= sm; }
            }
            for i in 0..seq { for d in 0..hd {
                let mut sm = 0.0f32;
                for j in 0..seq { sm += sc[i*seq+j]*v[j*kvd+kvh*hd+d]; }
                attn_out[i*dim+h*hd+d] = sm;
            }}
        }
        println!("Step 3: GQA Attention (CPU) [{:.1}s]", t0.elapsed().as_secs_f64());

        // Step 4: O proj
        let attn_proj = kn_tiled_gemm_opt(&mut ctx, &attn_out, seq, dim,
            "block_dim2048_o_proj", dim, kt, q_nt).await;
        println!("Step 4: O proj ({kt}Kt×{q_nt}Nt) [{:.1}s]", t0.elapsed().as_secs_f64());

        let h1 = vadd(&x_f32, &attn_proj);
        let normed2 = rmsnorm_cpu(&h1, &norm2_w, dim, eps);
        println!("Step 5-6: Residual + RMSNorm #2 [{:.1}s]", t0.elapsed().as_secs_f64());

        // Step 7: FFN Gate/Up
        let gate = kn_tiled_gemm_opt(&mut ctx, &normed2, seq, dim,
            "block_dim2048_ffn_gate", hidden, kt, gate_nt).await;
        println!("Step 7a: FFN gate ({kt}Kt×{gate_nt}Nt) [{:.1}s]", t0.elapsed().as_secs_f64());
        let up = kn_tiled_gemm_opt(&mut ctx, &normed2, seq, dim,
            "block_dim2048_ffn_up", hidden, kt, gate_nt).await;
        println!("Step 7b: FFN up ({kt}Kt×{gate_nt}Nt) [{:.1}s]", t0.elapsed().as_secs_f64());

        let ffn_mid = emul(&silu_cpu(&gate), &up);
        println!("Step 8: SiLU + mul [{:.1}s]", t0.elapsed().as_secs_f64());

        // Step 9: FFN Down
        let down = kn_tiled_gemm_opt(&mut ctx, &ffn_mid, seq, hidden,
            "block_dim2048_ffn_down", dim, down_kt, down_nt).await;
        println!("Step 9: FFN down ({down_kt}Kt×{down_nt}Nt) [{:.1}s]", t0.elapsed().as_secs_f64());

        let output = vadd(&h1, &down);
        println!("Step 10: Residual #2 [{:.1}s]", t0.elapsed().as_secs_f64());

        // 검증
        let expected = load_tiles_multi("block_dim2048_expected", seq, dim, kt);
        let elapsed = t0.elapsed().as_secs_f64();
        let mut max_diff: f32 = 0.0; let mut ref_max: f32 = 0.0;
        for i in 0..seq*dim {
            let d = (expected[i]-output[i]).abs(); if d>max_diff { max_diff=d; }
            if expected[i].abs()>ref_max { ref_max=expected[i].abs(); }
        }
        let rel_err = if ref_max>0.0 { max_diff/ref_max*100.0 } else { 0.0 };

        println!("\n=== Block dim2048 Full E2E (optimized) ===");
        println!("  dim={dim}, hidden={hidden}, n_heads={nh}, n_kv_heads={nkv}, head_dim={hd}");
        println!("  총 GEMM: 1088회 (HBM 입력 재사용 최적화 적용)");
        println!("  elapsed: {:.2}s", elapsed);
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);
        for i in 0..8 { println!("  [{}] exp={:.6} act={:.6} diff={:.6}",
            i, expected[i], output[i], (expected[i]-output[i]).abs()); }
        let tol = (ref_max*0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass {"PASS"} else {"FAIL"});
        assert!(pass, "block_dim2048 FAIL: max_diff={max_diff} > tol={tol}");
    }
}
