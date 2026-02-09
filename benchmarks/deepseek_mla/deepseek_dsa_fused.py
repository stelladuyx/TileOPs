import torch
from typing import Tuple, Dict, Any


class DeepSeekDSAFusedBenchmark:
    """Benchmark for FusedDeepSeekSparseAttentionFunc"""
    
    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        heads: int,
        dim: int,
        index_dim: int,
        topk: int,
        in_dtype: torch.dtype = torch.float16
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.index_dim = index_dim
        self.topk = topk
        self.in_dtype = in_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def gen_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate input tensors for DSA fused function"""
        return {
            "index_q": torch.randn(
                self.batch, self.seq_len, self.heads, self.index_dim,
                dtype=self.in_dtype, device=self.device
            ),
            "index_k": torch.randn(
                self.batch, self.seq_len_kv, self.index_dim,
                dtype=self.in_dtype, device=self.device
            ),
            "weights": torch.rand(
                self.batch, self.seq_len, self.heads,
                dtype=self.in_dtype, device=self.device
            ),
            "cu_seqlen_ks": torch.zeros(
                self.batch, self.seq_len,
                dtype=torch.int32, device=self.device
            ),
            "cu_seqlen_ke": torch.full(
                (self.batch, self.seq_len), self.seq_len_kv,
                dtype=torch.int32, device=self.device
            ),
            "starts": torch.zeros(
                self.batch, dtype=torch.int32, device=self.device
            ),
            "ends": torch.full(
                (self.batch,), self.seq_len_kv,
                dtype=torch.int32, device=self.device
            ),
            "query": torch.randn(
                self.batch, self.seq_len, self.heads, self.dim,
                dtype=self.in_dtype, device=self.device
            ),
            "kv_cache": torch.randn(
                self.batch, self.seq_len_kv, self.heads, self.dim + 8,
                dtype=self.in_dtype, device=self.device
            )
        }
    
    def check_fn(self, fn: Any, inputs: Dict[str, torch.Tensor], grad: bool = False) -> None:
        """Check function correctness"""
        output = fn(
            inputs["index_q"],
            inputs["index_k"],
            inputs["weights"],
            inputs["cu_seqlen_ks"],
            inputs["cu_seqlen_ke"],
            inputs["starts"],
            inputs["ends"],
            inputs["query"],
            inputs["kv_cache"]
        )
        
        # Validate output shape
        expected_shape = (self.batch, self.seq_len, self.heads, self.dim)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: {output.shape} != {expected_shape}"
        
        # Validate output dtype
        assert output.dtype == self.in_dtype, \
            f"Output dtype mismatch: {output.dtype} != {self.in_dtype}"
        
        # Validate no NaNs or Infs
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output dtype: {output.dtype}")
        print(f"✓ Output valid (no NaN/Inf)")