"""Apply pending SGLang fixes that the ROCm CI image does not ship yet.

The rocm/sgl-dev image bakes a fixed SGLang, so a fix that is still an open
upstream PR cannot reach the job container any other way. Each entry here names
the PR it mirrors and is idempotent: if the anchor text is gone the hunk is
assumed already present (image rebuilt with the fix) and is skipped, so this
script keeps working after the image catches up. Delete an entry once its PR is
in the image.

Usage: python tests/ci/apply_sglang_rocm_patches.py [--sglang-root PATH]
"""

import argparse
import ast
import sys
from pathlib import Path

# sgl-project/sglang#34016 -- weight checking for AITER-shuffled block FP8 weights.
# AITER pre-shuffles FP8 weights into a (16, 16) GEMM layout; the weight checker
# compared that raw layout against canonical row-major training weights and reported
# every FP8 tensor as mismatched. Tag the shuffled tensors and unshuffle before the
# comparison. Needed by test_deepseek_v4_flash_4layer_ci on ROCm.
PATCH_34016 = {
    "layers/quantization/fp8.py": [
        (
            "                t = shuffle_weight(layer.weight, (16, 16))\n"
            "                layer.weight.copy_(t)\n"
            "                del t",
            "                t = shuffle_weight(layer.weight, (16, 16))\n"
            "                layer.weight.copy_(t)\n"
            "                del t\n"
            "                layer.weight.is_shuffled = True",
        ),
        (
            "            t = shuffle_weight(layer.w2_weight, (16, 16))\n"
            "            layer.w2_weight.copy_(t)\n"
            "            del t\n"
            "        elif _is_cpu:",
            "            t = shuffle_weight(layer.w2_weight, (16, 16))\n"
            "            layer.w2_weight.copy_(t)\n"
            "            del t\n"
            "            layer.w13_weight.is_shuffled = True\n"
            "            layer.w2_weight.is_shuffled = True\n"
            "        elif _is_cpu:",
        ),
    ],
    "utils/weight_checker.py": [
        (
            "    comparable_cls: type[ComparableWeight]\n    scale_name: str\n",
            "    comparable_cls: type[ComparableWeight]\n    scale_name: str\n    is_shuffled: bool = False\n",
        ),
        (
            '    "_weight_fp32",\n',
            '    "_weight_fp32",\n    "cos_cache",\n    "sin_cache",\n',
        ),
        (
            "        own = {name for name, _ in module.named_parameters(recurse=False)}",
            "        own = dict(module.named_parameters(recurse=False))",
        ),
        (
            "                quantized_set[prefix + name] = QuantizedWeight(\n"
            "                    comparable_cls, prefix + scale\n"
            "                )",
            "                quantized_set[prefix + name] = QuantizedWeight(\n"
            "                    comparable_cls,\n"
            "                    prefix + scale,\n"
            '                    getattr(own[name], "is_shuffled", False),\n'
            "                )",
        ),
        (
            "                qw.comparable_cls(tensor, raw[qw.scale_name]),",
            "                qw.comparable_cls(\n"
            "                    tensor,\n"
            "                    raw[qw.scale_name],\n"
            "                    is_shuffled=qw.is_shuffled,\n"
            "                ),",
        ),
    ],
    "utils/weight_checker_comparator.py": [
        (
            "    def __init__(self, w_q: torch.Tensor, w_s: torch.Tensor):\n"
            "        self.w_q = w_q\n"
            "        self.w_s = w_s\n",
            "    def __init__(\n"
            "        self, w_q: torch.Tensor, w_s: torch.Tensor, is_shuffled: bool = False\n"
            "    ):\n"
            "        self.w_q = w_q\n"
            "        self.w_s = w_s\n"
            "        self.is_shuffled = is_shuffled\n",
        ),
        (
            '        return f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype})"',
            '        layout = " aiter_shuffled" if self.is_shuffled else ""\n'
            '        return f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype}{layout})"',
        ),
        (
            "    def _scale_and_block_size(self):",
            "    @staticmethod\n"
            "    def _unshuffle_aiter_weight(w_q: torch.Tensor) -> torch.Tensor:\n"
            '        """Restore AITER\'s (16, 16) GEMM layout to canonical row-major."""\n'
            "        shape = w_q.shape\n"
            "        n, k = shape[-2:]\n"
            '        assert n % 16 == 0 and k % 32 == 0, f"invalid AITER shuffled shape: {shape}"\n'
            "        return (\n"
            "            w_q.reshape(-1, n // 16, k // 32, 2, 16, 16)\n"
            "            .permute(0, 1, 4, 2, 3, 5)\n"
            "            .contiguous()\n"
            "            .reshape(shape)\n"
            "        )\n"
            "\n"
            "    def _scale_and_block_size(self):",
        ),
        (
            "            q, s_chunk = q.cuda(), s_chunk.cuda()",
            "            q, s_chunk = q.cuda(), s_chunk.cuda()\n"
            "            if self.is_shuffled:\n"
            "                q = self._unshuffle_aiter_weight(q)",
        ),
        (
            "        s, block_size = self._scale_and_block_size()\n"
            "        return block_quant_dequant(self.w_q, s, block_size, dtype=dtype)",
            "        s, block_size = self._scale_and_block_size()\n"
            "        w_q = self.w_q\n"
            "        if self.is_shuffled:\n"
            "            w_q = self._unshuffle_aiter_weight(w_q)\n"
            "        return block_quant_dequant(w_q, s, block_size, dtype=dtype)",
        ),
    ],
}

PATCHES = {"sglang#34016 aiter-shuffled block FP8 weight checking": PATCH_34016}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-root", default="/sgl-workspace/sglang/python/sglang/srt")
    args = ap.parse_args()
    root = Path(args.sglang_root)
    if not root.is_dir():
        print(f"sglang root not found, nothing to patch: {root}")
        return 0

    for label, files in PATCHES.items():
        applied = skipped = 0
        for rel, hunks in files.items():
            path = root / rel
            if not path.is_file():
                print(f"  {rel}: missing, skipping whole file")
                continue
            text = path.read_text()
            for old, new in hunks:
                # `new` contains `old` plus the additions, so "new already present"
                # is the reliable already-applied test. Checking `old` alone is not:
                # most anchors survive their own patch and would double-apply.
                if new in text or old not in text:
                    skipped += 1
                    continue
                text = text.replace(old, new, 1)
                applied += 1
            # Never leave a syntactically broken file behind.
            ast.parse(text)
            path.write_text(text)
        print(f"{label}: {applied} hunk(s) applied, {skipped} already present/absent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
