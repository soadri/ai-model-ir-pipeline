use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_intra_reduce_kernel::{
    A, R, pilot_golden_intra_reduce_kernel
};

#[tokio::main]
async fn main() { println!("intra_slice_reduce test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn intra_slice_reduce_sum() {
        let mut ctx = Context::acquire();

        // 테스트 입력: 1~16
        let x: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let expected_sum: f32 = x.iter().sum();  // = 136.0
        println!("입력: {:?}", &x);
        println!("기대 sum: {}", expected_sum);

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_golden_intra_reduce_kernel, (&mut ctx, &x_hbm)).await;
        let result: Vec<f32> = out_hbm.to_host::<m![R]>(&mut ctx.pdma).await.into_vec();

        println!("실제 sum: {}", result[0]);
        println!("오차:     {}", (expected_sum - result[0]).abs());

        assert!(
            (expected_sum - result[0]).abs() < 1e-3,
            "reduce_sum 불일치: expected={} got={}", expected_sum, result[0]
        );
        println!("✅ PASS: IntraSliceReduce f32 reduce_sum 성공");
    }
}
