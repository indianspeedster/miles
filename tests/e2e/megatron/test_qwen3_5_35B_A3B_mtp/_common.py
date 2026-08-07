"""Shared setup for Qwen3.5-35B-A3B MTP + speculative-v2 + R3 e2e cases.

Two cases differ in `enable_mtp_training` and `use_r3`:
- mtp1: MTP training on (1 layer + loss factor) + R3 on; the MTP/draft weights are synced.
- mtp0: MTP training off + R3 off. Whether the MTP layer is checked is a weight-check
  *selector* concern, not a skip-list one.

miles has no VLM/vision implementation on the training side, so Qwen3.5's `visual.*`
weights are never synced and must be excluded from the weight-equality check; each case
passes `check_weight_update_skip_list=("visual",)`.

Topology follows scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py (cp2/ep8 on 8 GPUs).
Spec (EAGLE) and spec-v2 (mamba scheduler) are on for the whole suite; R3 is per-case.
"""

import os
from dataclasses import dataclass

import torch

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3.5-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"

# Both ROCm branches below are workarounds for environment gaps, not for anything
# this suite is trying to test. Each is scoped to HIP so the CUDA cases -- the
# reference configuration -- keep running exactly as before.
_IS_HIP = torch.version.hip is not None


@dataclass
class CaseConfig:
    # Topology / GPU counts — explicit per case (each test file picks a shape that fits
    # the Qwen3.5 GatedDeltaNet backward on 8x80GB; see each file's CASE).
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    tp_size: int
    ep_size: int
    rollout_num_gpus_per_engine: int
    sglang_ep_size: int
    # Whether MTP training is enabled. On -> --enable-mtp-training --mtp-num-layers 1
    # (+ loss factor); off -> no MTP training args (effectively 0 MTP layers; note
    # `--enable-mtp-training --mtp-num-layers 0` would fail the arguments.py assert).
    enable_mtp_training: bool
    # Whether to enable R3 routing replay (--use-rollout-routing-replay).
    use_r3: bool
    max_tokens_per_gpu: int = 8192
    # Weight-check selector: "all" (target + draft) or "target" (target model only; use
    # when MTP training is off so the un-synced draft is not checked).
    check_weight_update_selector: str = "all"
    # Rollout weight-name substrings to exclude from the equality check (substring match;
    # mismatches become non-fatal). Cases pass ("visual",): miles has no VLM/vision
    # implementation on the training side, so those weights are never synced.
    check_weight_update_skip_list: tuple[str, ...] = ()


def prepare(case: CaseConfig) -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=case.num_gpus_per_node,
    )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    enable_eval = os.environ.get("MILES_TEST_ENABLE_EVAL", "0").lower() in ("1", "true", "yes")

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if enable_eval else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {case.tp_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {case.pp_size} "
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {case.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        # 0.6 (not 0.7): colocate leaves ~11GB of resident training memory on each GPU, so
        # sglang at 0.7 OOMs in the rollout MoE forward; 0.6 leaves headroom for both.
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-ep-size {case.sglang_ep_size} "
        "--sglang-max-running-requests 512 "
        # EAGLE speculative decoding (MTP draft)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        # spec v2: required to pair speculative decoding with radix cache on Qwen3.5MoE
        # (see scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py); also needs SGLANG_ENABLE_SPEC_V2=1.
        "--sglang-mamba-scheduler-strategy extra_buffer "
    )
    if case.use_r3:
        sglang_args += "--use-rollout-routing-replay "

    # When MTP training is off the rollout still runs EAGLE spec from the checkpoint
    # draft; those draft weights just never get synced (see the mtp0 case + skip-list).
    mtp_args = ""
    if case.enable_mtp_training:
        mtp_args = "--enable-mtp-training " "--mtp-num-layers 1 " "--mtp-loss-scaling-factor 0.2 "

    ci_args = "--ci-test "
    ci_args += f"--check-weight-update-selector {case.check_weight_update_selector} "
    if case.check_weight_update_skip_list:
        ci_args += "--check-weight-update-skip-list " + " ".join(case.check_weight_update_skip_list) + " "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {case.num_gpus_per_node} "
        "--colocate "
        # Megatron's flex dispatcher constructs _DeepepManager unconditionally
        # (token_dispatcher.py), and deep_ep is not installed in the rocm/sgl-dev
        # image, so flex raises ImportError at MoELayer build time on ROCm.
        # alltoall is the same fallback the other MoE suites reach for via their
        # use_deepep=False path. Drop this branch once the image ships deep_ep.
        f"--moe-token-dispatcher-type {'alltoall' if _IS_HIP else 'flex'} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{mtp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )
    return train_args


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    train_args = build_train_args(case, wandb_file=wandb_file)
    extra_env_vars = {"SGLANG_ENABLE_SPEC_V2": "1"}
    if _IS_HIP:
        # TODO(sglang): drop once RoutedExpertsCapturer stops reserving fused
        # shared-expert columns for models that append them outside the capture
        # site. It sizes its device buffer topk + model.num_fused_shared_experts
        # (state_capturer/routed_experts.py), but Qwen2/Qwen3.5 MoE builds TopK
        # without num_fused_shared_experts and appends the shared column after
        # select_experts returns (models/qwen2_moe.py _append_shared_to_topk_output).
        # The capture site therefore only ever sees topk columns, so the write is
        # 8 wide into a 9 wide slice and CUDA-graph capture dies. Only reachable
        # where shared-expert fusion is on, i.e. _use_aiter on ROCm -- which is
        # why CUDA never hits it. Turning aiter off also disables that fusion in
        # the rollout engine, so this costs MoE rollout throughput on ROCm.
        extra_env_vars["SGLANG_USE_AITER"] = "0"
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=extra_env_vars,
    )
