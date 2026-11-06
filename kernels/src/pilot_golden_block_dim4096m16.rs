use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::gemm_m16_k256_kernel::{
    M, K, N, gemm_m16_k256_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}
fn bf16_vec_from_bin(path: &str) -> Vec<bf16> {
    load_bf16_bin(path).chunks(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn bf16_to_f32(data: &[bf16]) -> Vec<f32> { data.iter().map(|v| f32::from(*v)).collect() }
fn pack_bf16_m16(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let pr=16; let pc=256;
    let mut buf = vec![0u8; pr*pc*2];
    for r in 0..rows.min(pr) { for c in 0..cols.min(pc) {
        let v = bf16::from_f32(data[r*cols+c]); let b = v.to_bits().to_le_bytes();
        let i = (r*pc+c)*2; buf[i]=b[0]; buf[i+1]=b[1];
    }} buf
}
fn extract_m16(p: &[f32], pc: usize, rows: usize, cols: usize) -> Vec<f32> {
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
async fn main() { println!("Block dim4096 M=16 K=256 E2E"); }

#[cfg(test)]
mod tests {
    use super::*;
    const PN: usize = 512;
    const GK: usize = 256;

    async fn npu_gemm(ctx: &mut Context, x: Vec<u8>, w: Vec<u8>) -> Vec<bf16> {
        let xh = HostTensor::<bf16, m![M,K]>::from_buf(x).to_hbm(&mut ctx.pdma).await;
        let wh = HostTensor::<bf16, m![N,K]>::from_buf(w).to_hbm(&mut ctx.pdma).await;
        let oh = launch(gemm_m16_k256_kernel, (ctx, &xh, &wh)).await;
        oh.to_host::<m![M,N]>(&mut ctx.pdma).await.into_vec()
    }

    async fn kn_tiled_gemm(ctx: &mut Context, input: &[f32], seq: usize, in_dim: usize,
                           prefix: &str, n_out: usize, kt: usize, n_tiles: usize,
                           t0: &std::time::Instant, label: &str, gc: &mut usize) -> Vec<f32> {
        let mut result = vec![0.0f32; seq * n_out];
        for nt in 0..n_tiles {
            let n_start = nt * 512;
            let n_cols = 512.min(n_out - n_start);
            let mut acc = vec![0.0f32; seq * n_cols];
            for t in 0..kt {
                let x_buf = {
                    let st = t*GK;
                    let mut tiled = vec![0.0f32; seq*GK];
                    for s in 0..seq { for c in 0..GK {
                        if st+c < in_dim { tiled[s*GK+c] = input[s*in_dim+st+c]; }
                    }}
                    pack_bf16_m16(&tiled, seq, GK)
                };
                let w_buf = load_bf16_bin(&format!("data/{prefix}_n{nt}_t{t}.bin"));
                let raw = npu_gemm(ctx, x_buf, w_buf).await;
                let p = bf16_to_f32(&raw);
                for s in 0..seq { for d in 0..n_cols { acc[s*n_cols+d] += p[s*PN+d]; } }
                *gc += 1;
            }
            for s in 0..seq { for d in 0..n_cols {
                result[s*n_out + n_start + d] = acc[s*n_cols + d];
            }}
            eprintln!("  [{:.1}s] {} Nt {}/{} (gemm={})", t0.elapsed().as_secs_f64(), label, nt+1, n_tiles, gc);
        }
        result
    }

    fn load_tiles(prefix: &str, seq: usize, dim: usize, kt: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq*dim];
        for t in 0..kt {
            let cols = GK.min(dim - t*GK);
            let tile = extract_m16(&bf16_to_f32(&bf16_vec_from_bin(
                &format!("data/{prefix}_t{t}.bin"))), GK, seq, cols);
            for s in 0..seq { for c in 0..cols { out[s*dim+t*GK+c] = tile[s*cols+c]; } }
        }
        out
    }

    fn load_norm(prefix: &str, dim: usize) -> Vec<f32> {
        let np = (dim+511)/512; let mut w = vec![0.0f32; dim];
        for p in 0..np {
            let part = bf16_to_f32(&bf16_vec_from_bin(&format!("data/{prefix}_p{p}.bin")));
            let s = p*512; let c = 512.min(dim-s);
            for i in 0..c { w[s+i] = part[i]; }
        }
        w
    }

    #[tokio::test]
    async fn block_dim4096m16_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();
        let dim=4096usize; let hidden=8192usize; let nh=8usize; let nkv=2usize;
        let hd=512usize; let seq=8usize; let nrep=nh/nkv; let eps=1e-5f32;
        let kvd=nkv*hd; // 1024
        let kt=dim/GK; // 16
        let q_nt=(dim+511)/512; // 8
        let kv_nt=(kvd+511)/512; // 2
        let gate_nt=(hidden+511)/512; // 16
        let down_kt=hidden/GK; // 32
        let down_nt=(dim+511)/512; // 8
        let mut gc = 0usize;

        let x_f32 = load_tiles("block_dim4096_x", seq, dim, kt);
        eprintln!("[{:.1}s] Input loaded", t0.elapsed().as_secs_f64());

        let norm1_w = load_norm("block_dim4096_input_norm", dim);
        let norm2_w = load_norm("block_dim4096_post_attn_norm", dim);

        let normed = rmsnorm_cpu(&x_f32, &norm1_w, dim, eps);
        eprintln!("[{:.1}s] Step 1: RMSNorm #1", t0.elapsed().as_secs_f64());

        // Q proj (16Kt×8Nt)
        let q = kn_tiled_gemm(&mut ctx, &normed, seq, dim,
            "block_dim4096_q_proj", dim, kt, q_nt, &t0, "Q", &mut gc).await;
        eprintln!("[{:.1}s] Step 2a: Q complete", t0.elapsed().as_secs_f64());

        // K proj (16Kt×2Nt — kvd=1024>512!)
        let k = kn_tiled_gemm(&mut ctx, &normed, seq, dim,
            "block_dim4096_k_proj", kvd, kt, kv_nt, &t0, "K", &mut gc).await;
        eprintln!("[{:.1}s] Step 2b: K complete", t0.elapsed().as_secs_f64());

        // V proj
        let v = kn_tiled_gemm(&mut ctx, &normed, seq, dim,
            "block_dim4096_v_proj", kvd, kt, kv_nt, &t0, "V", &mut gc).await;
        eprintln!("[{:.1}s] Step 2c: V complete", t0.elapsed().as_secs_f64());

        // GQA Attention (CPU)
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
        eprintln!("[{:.1}s] Step 3: GQA (CPU)", t0.elapsed().as_secs_f64());

        // O proj (16Kt×8Nt)
        let attn_proj = kn_tiled_gemm(&mut ctx, &attn_out, seq, dim,
            "block_dim4096_o_proj", dim, kt, q_nt, &t0, "O", &mut gc).await;

        let h1 = vadd(&x_f32, &attn_proj);
        let normed2 = rmsnorm_cpu(&h1, &norm2_w, dim, eps);
        eprintln!("[{:.1}s] Step 5-6: Residual + RMSNorm #2", t0.elapsed().as_secs_f64());

        // FFN gate/up (16Kt×16Nt)
        let gate = kn_tiled_gemm(&mut ctx, &normed2, seq, dim,
            "block_dim4096_ffn_gate", hidden, kt, gate_nt, &t0, "gate", &mut gc).await;
        let up = kn_tiled_gemm(&mut ctx, &normed2, seq, dim,
            "block_dim4096_ffn_up", hidden, kt, gate_nt, &t0, "up", &mut gc).await;

        let ffn_mid = emul(&silu_cpu(&gate), &up);
        eprintln!("[{:.1}s] Step 8: SiLU + mul", t0.elapsed().as_secs_f64());

        // FFN down (32Kt×8Nt)
        let down = kn_tiled_gemm(&mut ctx, &ffn_mid, seq, hidden,
            "block_dim4096_ffn_down", dim, down_kt, down_nt, &t0, "down", &mut gc).await;

        let output = vadd(&h1, &down);
        eprintln!("[{:.1}s] Step 10: Done (total GEMM={})", t0.elapsed().as_secs_f64(), gc);

        // 검증
        let expected = load_tiles("block_dim4096_expected", seq, dim, kt);
        let elapsed = t0.elapsed().as_secs_f64();
        let mut max_diff: f32 = 0.0; let mut ref_max: f32 = 0.0;
        for i in 0..seq*dim {
            let d = (expected[i]-output[i]).abs(); if d>max_diff { max_diff=d; }
            if expected[i].abs()>ref_max { ref_max=expected[i].abs(); }
        }
        let rel_err = if ref_max>0.0 { max_diff/ref_max*100.0 } else { 0.0 };

        println!("\n=== Block dim4096 M=16 K=256 Full E2E ===");
        println!("  dim=4096, hidden=8192, head_dim=512");
        println!("  총 GEMM: {}회 (M=16, K=256)", gc);
        println!("  elapsed: {:.2}s", elapsed);
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);
        for i in 0..8 { println!("  [{}] exp={:.6} act={:.6} diff={:.6}",
            i, expected[i], output[i], (expected[i]-output[i]).abs()); }
        let tol = (ref_max*0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass {"PASS"} else {"FAIL"});
        assert!(pass, "FAIL: max_diff={max_diff} > tol={tol}");
    }
}
