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
async fn main() { println!("Block dim256 E2E"); }

#[cfg(test)]
mod tests {
    use super::*;
    async fn npu_gemm(ctx: &mut Context, x: Vec<u8>, w: Vec<u8>) -> Vec<bf16> {
        let xh = HostTensor::<bf16, m![M,K]>::from_buf(x).to_hbm(&mut ctx.pdma).await;
        let wh = HostTensor::<bf16, m![N,K]>::from_buf(w).to_hbm(&mut ctx.pdma).await;
        let oh = launch(pilot_golden_ffn_gate_kernel, (ctx, &xh, &wh)).await;
        oh.to_host::<m![M,N]>(&mut ctx.pdma).await.into_vec()
    }
    async fn k_tiled_gemm(ctx: &mut Context, xt: &[Vec<u8>], wt: &[Vec<u8>], seq: usize, nout: usize) -> Vec<f32> {
        let pn = 512usize; let nt = xt.len();
        let mut acc = vec![0.0f32; seq*nout];
        for t in 0..nt {
            let raw = npu_gemm(ctx, xt[t].clone(), wt[t].clone()).await;
            let p = bf16_to_f32(&raw);
            for s in 0..seq { for d in 0..nout { acc[s*nout+d] += p[s*pn+d]; } }
        } acc
    }
    fn tile_input(data: &[f32], seq: usize, fc: usize, ti: usize, tk: usize) -> Vec<u8> {
        let st = ti*tk; let mut t = vec![0.0f32; seq*tk];
        for s in 0..seq { for c in 0..tk { if st+c < fc { t[s*tk+c] = data[s*fc+st+c]; } } }
        pack_bf16(&t, seq, tk, 512, 64)
    }
    fn load_tiles(prefix: &str, n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|t| load_bf16_bin(&format!("data/{prefix}_t{t}.bin"))).collect()
    }

    #[tokio::test]
    async fn block_dim256_end_to_end() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();
        let dim=256; let hidden=512; let nh=8; let nkv=2; let hd=32; let seq=8;
        let nrep = nh/nkv; let eps=1e-5f32; let kvd = nkv*hd; let kt = dim/64;

        // Load input [8,256] from 4 tiles
        let mut x_f32 = vec![0.0f32; seq*dim];
        for t in 0..kt {
            let tile = extract_region(&bf16_to_f32(&bf16_vec_from_bin(&format!("data/block_dim256_x_t{t}.bin"))), 64, seq, 64);
            for s in 0..seq { for c in 0..64 { x_f32[s*dim+t*64+c] = tile[s*64+c]; } }
        }
        println!("Input x[0:4]: {:.4?}", &x_f32[..4]);

        let norm1_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim256_input_norm.bin"));
        let norm2_w = bf16_to_f32(&bf16_vec_from_bin("data/block_dim256_post_attn_norm.bin"));

        // Step 1: RMSNorm #1
        let normed = rmsnorm_cpu(&x_f32, &norm1_w, dim, eps);
        println!("Step 1: RMSNorm #1");

        // Step 2: Q/K/V projection (4-tile each)
        let nt: Vec<Vec<u8>> = (0..kt).map(|t| tile_input(&normed, seq, dim, t, 64)).collect();
        let q = k_tiled_gemm(&mut ctx, &nt, &load_tiles("block_dim256_q_proj", kt), seq, dim).await;
        println!("Step 2a: Q proj (NPU {kt}-tile). Q[0:4]={:.4?}", &q[..4]);
        let k = k_tiled_gemm(&mut ctx, &nt, &load_tiles("block_dim256_k_proj", kt), seq, kvd).await;
        println!("Step 2b: K proj (NPU {kt}-tile). K[0:4]={:.4?}", &k[..4]);
        let v = k_tiled_gemm(&mut ctx, &nt, &load_tiles("block_dim256_v_proj", kt), seq, kvd).await;
        println!("Step 2c: V proj (NPU {kt}-tile). V[0:4]={:.4?}", &v[..4]);

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
                for v in row.iter_mut() { *v = (*v-mx).exp(); sm += *v; }
                for v in row.iter_mut() { *v /= sm; }
            }
            for i in 0..seq { for d in 0..hd {
                let mut sm = 0.0f32;
                for j in 0..seq { sm += sc[i*seq+j]*v[j*kvd+kvh*hd+d]; }
                attn_out[i*dim+h*hd+d] = sm;
            }}
        }
        println!("Step 3: GQA Attention (CPU)");

        // Step 4: O projection (4-tile)
        let aot: Vec<Vec<u8>> = (0..kt).map(|t| tile_input(&attn_out, seq, dim, t, 64)).collect();
        let attn_proj = k_tiled_gemm(&mut ctx, &aot, &load_tiles("block_dim256_o_proj", kt), seq, dim).await;
        println!("Step 4: O proj (NPU {kt}-tile)");

        // Step 5: Residual #1
        let h1 = vadd(&x_f32, &attn_proj);

        // Step 6: RMSNorm #2
        let normed2 = rmsnorm_cpu(&h1, &norm2_w, dim, eps);

        // Step 7: FFN Gate + Up (4-tile each)
        let n2t: Vec<Vec<u8>> = (0..kt).map(|t| tile_input(&normed2, seq, dim, t, 64)).collect();
        let gate = k_tiled_gemm(&mut ctx, &n2t, &load_tiles("block_dim256_ffn_gate", kt), seq, hidden).await;
        let up = k_tiled_gemm(&mut ctx, &n2t, &load_tiles("block_dim256_ffn_up", kt), seq, hidden).await;
        println!("Step 7: FFN gate/up (NPU {kt}-tile each)");

        // Step 8: SiLU + mul
        let ffn_mid = emul(&silu_cpu(&gate), &up);

        // Step 9: FFN Down (8-tile)
        let hkt = hidden/64;
        let mt: Vec<Vec<u8>> = (0..hkt).map(|t| tile_input(&ffn_mid, seq, hidden, t, 64)).collect();
        let down = k_tiled_gemm(&mut ctx, &mt, &load_tiles("block_dim256_ffn_down", hkt), seq, dim).await;
        println!("Step 9: FFN down (NPU {hkt}-tile)");

        // Step 10: Residual #2
        let output = vadd(&h1, &down);
        println!("Step 10: Residual #2");

        // 검증
        let mut expected = vec![0.0f32; seq*dim];
        for t in 0..kt {
            let tile = extract_region(&bf16_to_f32(&bf16_vec_from_bin(&format!("data/block_dim256_expected_t{t}.bin"))), 64, seq, 64);
            for s in 0..seq { for c in 0..64 { expected[s*dim+t*64+c] = tile[s*64+c]; } }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let mut max_diff: f32 = 0.0; let mut ref_max: f32 = 0.0;
        for i in 0..seq*dim {
            let d = (expected[i]-output[i]).abs(); if d>max_diff { max_diff=d; }
            if expected[i].abs()>ref_max { ref_max=expected[i].abs(); }
        }
        let rel_err = if ref_max>0.0 { max_diff/ref_max*100.0 } else { 0.0 };

        println!("\n=== Block dim256 Full E2E ===");
        println!("  dim={dim}, hidden={hidden}, n_heads={nh}, n_kv_heads={nkv}, head_dim={hd}");
        println!("  NPU GEMM: Q/K/V(4t×3)+O(4t)+gate/up(4t×2)+down(8t) = 32회");
        println!("  elapsed: {:.2}s", elapsed);
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);
        for i in 0..8 { println!("  [{}] exp={:.6} act={:.6} diff={:.6}", i, expected[i], output[i], (expected[i]-output[i]).abs()); }
        let tol = (ref_max*0.15).max(0.5);
        let pass = max_diff <= tol;
        println!("  tol={:.4} → {}", tol, if pass {"PASS"} else {"FAIL"});
        assert!(pass, "block_dim256 FAIL: max_diff={max_diff} > tol={tol}");
    }
}
