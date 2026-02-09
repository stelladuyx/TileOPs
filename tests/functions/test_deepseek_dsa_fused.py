import pytest
import torch

from benchmarks import DeepSeekDSAFusedBenchmark
from top.functions import FusedDeepSeekSparseAttentionFunc
from top.utils import str2dtype


@pytest.mark.parametrize(
    ("batch, seq_len, seq_len_kv, heads, dim, index_dim, topk, in_dtype, tune"),
    [
        (2, 512, 8192, 32, 128, 64, 256, torch.float16, False),
        (2, 512, 8192, 32, 128, 64, 512, torch.bfloat16, False),
        (1, 1024, 4096, 16, 256, 128, 512, torch.float32, False),
        (4, 256, 16384, 64, 64, 32, 128, torch.float16, False),
    ],
)
def test_deepseek_dsa_fused(batch, seq_len, seq_len_kv, heads, dim, index_dim, topk, in_dtype, tune=False):
    """Test FusedDeepSeekSparseAttentionFunc with various configurations"""
    
    # Initialize function
    fn = FusedDeepSeekSparseAttentionFunc(
        seq_len_kv=seq_len_kv,
        index_dim=index_dim,
        seq_len=seq_len,
        heads=heads,
        batch=batch,
        topk=topk,
        dim=dim,
        dim_tail=8,  # Default tail dimension
        stride_kv=1,
        group_kv=1,
        q_start_index_s=0,
        quant_in_dtype=in_dtype,
        clean_logits=True,
        in_dtype=str2dtype(str(in_dtype)),
        out_dtype="int32",
        dsa_dtype=in_dtype,
        tune=tune
    )
    
    # Initialize benchmark
    benchmark = DeepSeekDSAFusedBenchmark(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        heads=heads,
        dim=dim,
        index_dim=index_dim,
        topk=topk,
        in_dtype=in_dtype
    )
    
    # Generate inputs
    inputs = benchmark.gen_inputs()
    
    try:
        print("Testing FusedDeepSeekSparseAttentionFunc...")
        benchmark.check_fn(fn, inputs, grad=False)
        print("✅ FusedDeepSeekSparseAttentionFunc test passed")
    except Exception as e:
        print(f"❌ FusedDeepSeekSparseAttentionFunc test failed: {e}")
        raise


if __name__ == "__main__":
    test_deepseek_dsa_fused(2, 512, 8192, 32, 128, 64, 256, torch.float16, False)
    test_deepseek_dsa_fused(2, 512, 8192, 32, 128, 64, 512, torch.bfloat16, False)
    test_deepseek_dsa_fused(1, 1024, 4096, 16, 256, 128, 512, torch.float32, False)
    test_deepseek_dsa_fused(4, 256, 16384, 64, 64, 32, 128, torch.float16, False)