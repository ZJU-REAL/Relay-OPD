# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import socket
import struct
import threading
import time
from collections import defaultdict
from pprint import pprint
from typing import Any, Callable, Optional

import ray
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from verl.plugin.platform import get_platform
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_vllm_profiler_args
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, qwen2_5_vl_dedup_image_tokens, run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    extract_prompt_logprobs,
    get_vllm_max_lora_rank,
)

_VLLM_VERSION = version.parse(vllm.__version__)
_RESET_PREFIX_CACHE_KWARGS = {}
if _VLLM_VERSION >= version.parse("0.13.0"):
    _RESET_PREFIX_CACHE_KWARGS["reset_connector"] = True


if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

    elif _VLLM_VERSION >= version.parse("0.13.0"):
        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    else:
        get_encoding = None

    if get_encoding is not None and os.getenv("VERL_USE_GPT_OSS", "0") == "1":
        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _apply_opd_speculative_decode_if_needed() -> None:
    """Apply OPD speculative-decoding patches inside the rollout server.

    This is intentionally called after the Ray actor process has started, so
    VERL_OPD_VLLM_PATCH_DEFER=1 can avoid importing/patching vLLM from
    sitecustomize during generic Ray worker startup.
    """
    if os.environ.get("VERL_OPD_VLLM_PATCH") != "1":
        return
    # Ray/TaskRunner startup can defer sitecustomize patching. Once we are
    # inside the actor rollout vLLM server, turn the defer gate off before
    # vLLM launches worker subprocesses so those children patch via
    # sitecustomize at interpreter startup too.
    os.environ["VERL_OPD_VLLM_PATCH_DEFER"] = "0"
    from speculative_decode import apply_patches

    apply_patches()


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
            is_reward_model (bool): whether this server belongs to a reward model replica.
            is_teacher_model (bool): whether this server belongs to a distillation teacher replica.
        """
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)
        # Forward the Ray job id into the vLLM worker subprocess so the
        # colocated weight-transfer IPC socket path is unique per Ray job.
        # Without this, two concurrent verl jobs on the same node both bind
        # the same /tmp/rl-colocate-zmq-replica-0-rank-0.sock and one fails
        # with EADDRINUSE; a stale socket from a crashed run trips the same
        # error on restart.
        os.environ["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()

        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.is_reward_model = is_reward_model
        self.is_teacher_model = is_teacher_model
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        self._preprocess_engine_kwargs(engine_kwargs)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"

        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + (self.config.get("seed") or 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # update profiler args
        profiler_args = build_vllm_profiler_args(
            self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
        )
        if _VLLM_VERSION >= version.parse("0.13.0"):
            # vLLM >= 0.13.0 supports profiler config via CLI args; env vars still work but will be deprecated
            args.update(profiler_args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # Patched speculative decoding only applies to the actor rollout server.
        opd_speculative_enabled = (
            os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1"
            and not self.is_teacher_model
            and not self.is_reward_model
        )
        if opd_speculative_enabled:
            _apply_opd_speculative_decode_if_needed()
            args["speculative_config"] = {
                "method": "draft_model",
                "model": os.environ["VERL_OPD_DRAFT_MODEL"],
                "num_speculative_tokens": int(os.environ.get("VERL_OPD_NUM_SPECULATIVE_TOKENS", "1")),
                "draft_tensor_parallel_size": int(os.environ.get("VERL_OPD_DRAFT_TENSOR_PARALLEL_SIZE", "1")),
            }
            sock_path = f"/dev/shm/verl_opd_mask_{os.getpid()}.sock"
            self._opd_mask_ipc = OpdMaskIpcServer(sock_path)
            self._opd_mask_ipc.start()
            os.environ["VERL_OPD_MASK_IPC_SOCKET"] = sock_path
            logger.warning("[opd-rollout] mask IPC server started: %s", sock_path)
        elif self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            speculative_config = {
                "method": self.config.mtp.method,
                "num_speculative_tokens": self.config.mtp.num_speculative_tokens,
            }
            args["speculative_config"] = speculative_config

        if self.config.data_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to "
                f"dp_size_local ({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"
            )
            dp_args = {
                "data_parallel_size": self.config.data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._dp_rpc_port,
            }
            args.update(dp_args)

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        _opd_target_model_path = os.environ.get("VERL_OPD_TARGET_MODEL") if opd_speculative_enabled else None
        _serve_model_path = _opd_target_model_path or self.model_config.local_path
        server_args = ["serve", _serve_model_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        build_app_kwargs: dict[str, Any] = {}
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            build_app_kwargs["supported_tasks"] = supported_tasks
        # vLLM >= 0.20.0 requires `model_config` to register pooling API routes
        # (e.g. ``/classify``, ``/embed``); see
        # ``register_pooling_api_routers`` in vllm/entrypoints/pooling/factories.py
        # which short-circuits when ``model_config`` is ``None``.
        if "model_config" in build_app_sig.parameters:
            build_app_kwargs["model_config"] = engine_client.model_config
        app = build_app(args, **build_app_kwargs)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        prompt_ids = normalize_token_ids(prompt_ids)
        if (
            os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1"
            and not self.is_teacher_model
            and not self.is_reward_model
            and not hasattr(self, "_opd_mask_ipc")
        ):
            raise RuntimeError(
                "VERL_OPD_SPECULATIVE_ENABLE=1 but OPD mask IPC server is not initialized "
                f"for rollout request_id={request_id}. This would produce response tokens without relay_teacher_mask."
            )

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound. vLLM v0.20+ rejects `max_tokens < 1`
        # (see vllm.sampling_params.SamplingParams._verify_args), so we require at
        # least one token of headroom to be able to generate at all.
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 1:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) leaves no room to generate within the "
                f"model's maximum context length ({self.config.max_model_len}); need at least "
                f"1 token of headroom."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_tokens to the valid range [1, max_possible_tokens]. The lower bound
        # is 1 because vLLM v0.20+ raises VLLMValidationError when max_tokens < 1.
        max_tokens = max(1, min(max_tokens, max_possible_tokens))

        assert 1 <= max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} not in valid range [1, {max_possible_tokens}]"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data

        prompt_kwargs = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}
        if mm_processor_kwargs:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        try:
            prompt = TokensPrompt(**prompt_kwargs)
        except TypeError:
            prompt = prompt_kwargs

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        # Handle abort case: when the request is aborted by pause_generation(abort),
        # outputs may be empty. Return empty results with stop_reason="aborted"
        # instead of crashing with "IndexError: list index out of range".
        if not final_res.outputs:
            logger.warning(
                "[rollout] empty vLLM output: request_id=%s global_steps=%s prompt_len=%s",
                request_id,
                self.global_steps,
                len(prompt_ids),
            )
            extra_fields = {
                "request_id": request_id,
                "global_steps": self.global_steps,
                "stop_reason": "aborted",
                "trigger_stopped": False,
                "trigger_stop_pos": None,
            }
            if os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1":
                extra_fields["relay_teacher_mask"] = []
            return TokenOutput(
                token_ids=[],
                log_probs=[] if sampling_params.logprobs is not None else None,
                routed_experts=None,
                stop_reason="aborted",
                extra_fields=extra_fields,
            )

        extra_fields = {
            "request_id": request_id,
            "global_steps": self.global_steps,
            "trigger_stopped": False,
            "trigger_stop_pos": None,
        }
        include_token_logprobs = os.environ.get("VERL_DISTILLATION_DUAL_LOGPROBS") == "1"
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            result_dict=extra_fields,
            prompt_token_ids=prompt_ids if include_token_logprobs else None,
            include_token_logprobs=include_token_logprobs,
        )
        raw_token_ids = list(final_res.outputs[0].token_ids)
        token_ids = raw_token_ids
        trigger_stop_event = None
        if os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1" and hasattr(self, "_opd_mask_ipc"):
            trigger_stop_event = self._opd_mask_ipc.pop_stop_event(request_id, timeout_s=0.05)
        if trigger_stop_event is not None:
            try:
                truncate_pos = int(trigger_stop_event["response_pos"])
                internal_stop_id = int(trigger_stop_event["internal_stop_token_id"])
                if 0 <= truncate_pos < len(raw_token_ids) and int(raw_token_ids[truncate_pos]) == internal_stop_id:
                    token_ids = raw_token_ids[:truncate_pos]
                    trigger_stop_event = dict(trigger_stop_event)
                    trigger_stop_event.update({
                        "raw_token_len": len(raw_token_ids),
                        "stripped_token_len": len(token_ids),
                        "forced_stop_token_removed": True,
                    })
                    if len(token_ids) == 0:
                        logger.warning(
                            "[opd-rollout] trigger stop stripped response to empty: "
                            "request_id=%s global_steps=%s event=%s raw_token_ids_head=%s",
                            request_id,
                            self.global_steps,
                            trigger_stop_event,
                            raw_token_ids[:5],
                        )
                elif truncate_pos == len(raw_token_ids):
                    trigger_stop_event = dict(trigger_stop_event)
                    trigger_stop_event.update({
                        "raw_token_len": len(raw_token_ids),
                        "stripped_token_len": len(token_ids),
                        "forced_stop_token_removed": True,
                        "already_stripped": True,
                    })
                else:
                    logger.warning(
                        "[opd-rollout] trigger stop event did not match final token ids: "
                        "request_id=%s pos=%s stop_id=%s raw_len=%s tail=%s",
                        request_id,
                        truncate_pos,
                        internal_stop_id,
                        len(raw_token_ids),
                        raw_token_ids[max(0, truncate_pos - 2): truncate_pos + 3],
                    )
                    trigger_stop_event = None
            except Exception:
                logger.exception("[opd-rollout] failed to apply trigger stopping event for request_id=%s", request_id)
                trigger_stop_event = None
        log_probs = None
        if sampling_params.logprobs is not None:
            raw_log_probs = [
                logprobs[raw_token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)
            ]
            log_probs = raw_log_probs[: len(token_ids)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future
        extra_fields["stop_reason"] = stop_reason
        if len(token_ids) == 0:
            logger.warning(
                "[rollout] empty token_ids after postprocess: request_id=%s global_steps=%s "
                "finish_reason=%s trigger_stopped=%s stop_event=%s raw_token_len=%s",
                request_id,
                self.global_steps,
                finish_reason,
                trigger_stop_event is not None,
                trigger_stop_event,
                len(raw_token_ids),
            )

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        # Export token-source metadata from the vLLM worker subprocess.
        if os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1" and hasattr(self, "_opd_mask_ipc"):
            mask, action_ids, action_logps, action_mask = self._opd_mask_ipc.pop_with_actions(
                request_id, expected_len=len(token_ids), token_ids=token_ids
            )
            if mask is None:
                raise RuntimeError(
                    f"relay_teacher_mask missing for request_id={request_id}; "
                    "IPC failed or the vLLM patch did not emit an absolute-position mask."
                )
            if len(mask) != len(token_ids):
                raise RuntimeError(
                    f"relay_teacher_mask length mismatch after alignment for request_id={request_id}: "
                    f"got {len(mask)}, expected {len(token_ids)}, diff={len(token_ids) - len(mask)}"
                )
            extra_fields["relay_teacher_mask"] = mask
            if action_ids is not None and action_logps is not None and action_mask is not None:
                extra_fields["relay_student_action_ids"] = action_ids
                extra_fields["relay_student_action_teacher_logprobs"] = action_logps
                extra_fields["relay_student_action_mask"] = action_mask
        if trigger_stop_event is not None:
            extra_fields["trigger_stopped"] = True
            extra_fields["trigger_stop_pos"] = trigger_stop_event.get("response_pos")

        # Re-key backend spec-decoding stats to the rollout-common names.
        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            if final_res.metrics is None or final_res.metrics.request_spec_decode_stats is None:
                raise RuntimeError("vLLM MTP rollout requires request_spec_decode_stats; set disable_log_stats=False.")
            spec_decode_stats = final_res.metrics.request_spec_decode_stats
            extra_fields["spec_num_draft_tokens"] = spec_decode_stats.num_draft_tokens
            extra_fields["spec_num_accepted_tokens"] = spec_decode_stats.num_accepted_tokens
            extra_fields["spec_num_verify_steps"] = spec_decode_stats.num_verify_steps
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wake_up(self, tags: list[str] | None = None):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # engine.wake_up() broadcasts via the DP coordinator to ALL EngineCore
            # processes across all DP shards (unlike collective_rpc which only reaches
            # TP workers within a single shard).
            await self.engine.wake_up(tags=tags or self._get_wake_up_tags())
            await self.engine.reset_prefix_cache(**_RESET_PREFIX_CACHE_KWARGS)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            # reset_connector=True drops any attached external KV store
            # (e.g. MooncakeStoreConnector) whose entries were computed
            # against the previous weights. No-op success when no connector
            # is configured (vLLM scheduler treats it as such).
            await self.engine.reset_prefix_cache(**_RESET_PREFIX_CACHE_KWARGS)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self._sleep_hybrid()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            # reset_connector=True drops any attached external KV store
            # (e.g. MooncakeStoreConnector) whose entries were computed
            # against the previous model weights. With no connector it
            # is a no-op success, so we can pass it unconditionally.
            await self.engine.reset_prefix_cache(**_RESET_PREFIX_CACHE_KWARGS)

            if _VLLM_VERSION >= version.parse("0.9.0"):
                await self.engine.reset_mm_cache()
            if _VLLM_VERSION >= version.parse("0.16.0"):
                await self.engine.reset_encoder_cache()

    async def release_kv_cache(self):
        """Release only kv_cache GPU memory, keeping model weights intact.
        # TODO: support true release of kv_cache
        """
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

    async def resume_kv_cache(self):
        """Restore kv_cache GPU memory after a weight sync. Counterpart to release_kv_cache()."""
        if self.node_rank != 0:
            return

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.stop_profile()

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        On vLLM >= 0.12.0, uses AsyncLLM.pause_generation() to abort in-flight
        requests, drain, and clear caches. The engine remains paused after this
        call — use resume_generation() to accept new requests (e.g. before
        validation).

        On vLLM < 0.12.0, manually aborts each request and resets prefix cache.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            if _VLLM_VERSION >= version.parse("0.12.0"):
                # Snapshot request IDs before pausing for reporting
                request_ids = list(self.engine.output_processor.request_states.keys())

                # pause_generation with wait_for_inflight_requests=False will:
                # 1. Set engine to paused state (blocks new generate calls)
                # 2. Abort all in-flight requests
                # 3. Wait for requests to drain
                # 4. Clear prefix and mm caches if clear_cache=True.
                #    EngineCore._reset_caches defaults reset_connector=True
                #    on this path, so any attached external KV store (e.g.
                #    MooncakeStoreConnector) is invalidated along with the
                #    local prefix cache — RL-correct hard-reset at every
                #    weight update boundary, no extra kwargs needed.
                await self.engine.pause_generation(
                    wait_for_inflight_requests=False,
                    clear_cache=reset_prefix_cache,
                )
            else:
                # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
                request_states_snapshot = list(self.engine.output_processor.request_states.items())
                request_ids = [req_id for req_id, _ in request_states_snapshot]

                if not request_ids:
                    return {"aborted_count": 0, "request_ids": []}

                # For each request, create an abort output and put it to its queue
                # This allows the generator to receive the aborted result
                from vllm.v1.engine import FinishReason

                for _, req_state in request_states_snapshot:
                    request_output = req_state.make_request_output(
                        [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                    )
                    req_state.queue.put(request_output)

                # Abort requests in the output processor and engine core
                self.engine.output_processor.abort_requests(request_ids)
                await self.engine.engine_core.abort_requests_async(request_ids)

                # Try to reset prefix cache to ensure clean state
                if reset_prefix_cache:
                    await self.clear_kv_cache()
                    logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def resume_generation(self):
        """Resume generation after abort_all_requests (pause_generation).

        Only effective on vLLM >= 0.12.0 where pause_generation is used.
        No-op on older versions.
        """
        if self.node_rank != 0:
            return
        if _VLLM_VERSION >= version.parse("0.12.0"):
            await self.engine.resume_generation()

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        """Initialise config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(config)

    def _init_model_config(self, model_config):
        """Initialise model_config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

    def _validate_configs(self) -> None:
        """Validate config/model_config after initialisation."""
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Called at the end of __init__. Default logs server metadata."""
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def _get_engine_kwargs_key(self) -> str:
        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
        return "vllm"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Mutate engine_kwargs in-place before the CLI args dict is built. No-op by default."""
        pass

    def _get_override_generation_config(self) -> dict:
        """Return the override_generation_config dict."""
        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        return dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )

    def _apply_quantization(self) -> tuple[Optional[str], dict]:
        """Process quantization config. Returns (quantization_str, hf_overrides)."""
        quantization = self.config.quantization
        hf_overrides = {}

        if is_torch_npu_available(check_device=False):
            from verl.utils.vllm.npu_vllm_patch import check_vllm_ascend_before_server_launch

            check_vllm_ascend_before_server_launch()

        # Handle QAT (Quantization-Aware Training) configuration
        qat_config_dict = getattr(self.config, "qat", {}) or {}
        if qat_config_dict.get("enable", False):
            from verl.utils.qat import QATConfig, load_quantization_config

            qat_config = QATConfig(**qat_config_dict)
            quantization_config_dict = load_quantization_config(qat_config)
            quant_method = quantization_config_dict.get("quant_method", None)

            if quant_method == "modelopt":
                from verl.utils.modelopt import apply_modelopt_nvfp4_patches

                apply_modelopt_nvfp4_patches()
                quantization = "modelopt"
            elif quant_method == "compressed-tensors":
                from verl.utils.qat import apply_qat_patches

                apply_qat_patches()
                quantization = "compressed-tensors"
            else:
                raise ValueError(f"Unsupported quant_method: {quant_method}")

            logger.info(f"QAT quantization config injected (quant_method={quant_method})")
            hf_overrides["quantization_config"] = quantization_config_dict
        elif quantization is not None:
            # Handle other quantization methods (fp8, torchao)
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao", "ascend"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                # Ignore MoE router layers for FP8 quantization
                all_mlp_gate_layers = []
                for layer in range(self.model_config.hf_config.num_hidden_layers):
                    all_mlp_gate_layers.append(f"model.layers.{layer}.mlp.gate")

                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                    "ignored_layers": all_mlp_gate_layers,
                }
                hf_overrides["quantization_config"] = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
                # for subprocesses patching
                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"

        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        return quantization, hf_overrides

    def _get_worker_extension_cls(self) -> str:
        """Return the fully-qualified colocate worker extension class name."""
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        """Return the list of CLI command modules used for argument parsing."""
        return [vllm.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        """Return the description string for the CLI argument parser."""
        return "vLLM CLI"

    def _get_wake_up_tags(self) -> list[str]:
        """Return the tags passed to engine.wake_up(). Default includes kv_cache."""
        return ["kv_cache", "weights"]

    async def _sleep_hybrid(self):
        """HYBRID sleep: lora adapters only need level=1; full weights need level=2.

        Uses engine.sleep() instead of engine.collective_rpc("sleep") to ensure
        that sleep is properly propagated to all data-parallel worker processes.
        collective_rpc only reaches the TP workers within a single DP shard,
        leaving other DP shards' weights unreleased, which causes OOM during
        FSDP training backward when DP > 1.
        """
        # lora only update adapter weights, so set sleep level to 1
        # vllm_ascend not support sleep_level now. Enabling EP during training may lead to accuracy issues.
        # OPD speculative decoding has target teacher + draft student;
        # level=2 discards weights without CPU backup, but only draft gets
        # actor sync — teacher has no rebuild source. Force level=1 so both
        # models are backed up to CPU and restored on wake_up.
        opd_speculative_enabled = (
            os.environ.get("VERL_OPD_SPECULATIVE_ENABLE") == "1"
            and not self.is_teacher_model
            and not self.is_reward_model
        )
        if self.lora_as_adapter or is_torch_npu_available(check_device=False) or opd_speculative_enabled:
            sleep_level = 1
        else:
            sleep_level = 2
        await self.engine.sleep(level=sleep_level)
        if _VLLM_VERSION >= version.parse("0.17.0"):
            await self.engine.reset_encoder_cache()


class OpdMaskIpcServer:
    """Receive absolute-position token-source metadata from vLLM workers."""

    def __init__(self, sock_path: str):
        self.path = sock_path
        self.mask_store: dict[str, list[bool]] = defaultdict(list)
        self.token_store: dict[str, list[int]] = defaultdict(list)
        self.position_store: dict[str, list[int]] = defaultdict(list)
        self.action_store: dict[str, list[int]] = defaultdict(list)
        self.action_logp_store: dict[str, list[float]] = defaultdict(list)
        self.stop_event_store: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(self.path)
        self.sock.settimeout(1.0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                payload, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            parts = payload.split(b"\0", 1)
            if len(parts) != 2:
                continue
            rid = parts[0].decode("utf-8")
            stop_event = self._decode_stop_event_payload(parts[1])
            if stop_event is not None:
                with self.lock:
                    self.stop_event_store[rid].append(stop_event)
                continue
            mask, tokens, positions, actions, action_logps = self._decode_payload(parts[1])
            if not mask:
                continue
            with self.lock:
                self.mask_store[rid].extend(mask)
                if tokens is not None:
                    self.token_store[rid].extend(tokens)
                if positions is not None:
                    self.position_store[rid].extend(positions)
                if actions is not None:
                    self.action_store[rid].extend(actions)
                if action_logps is not None:
                    self.action_logp_store[rid].extend(action_logps)

    @staticmethod
    def _decode_stop_event_payload(body: bytes) -> dict[str, Any] | None:
        if not body.startswith(b"OPS1"):
            return None
        try:
            event = json.loads(body[4:].decode("utf-8"))
        except Exception:
            logger.exception("[opd-rollout] failed to decode trigger stop IPC payload")
            return None
        if not isinstance(event, dict):
            return None
        return event

    @staticmethod
    def _decode_payload(
        body: bytes,
    ) -> tuple[list[bool], list[int] | None, list[int] | None, list[int] | None, list[float] | None]:
        # OPD3 payload from opd.patches.vllm.speculative_decode:
        #   b"OPD3" + uint8(n) + n mask bytes
        #   + n int32 emitted ids + n int32 positions
        #   + n int32 PG action ids + n float32 teacher logp(action).
        if body.startswith(b"OPD3") and len(body) >= 5:
            n = body[4]
            expected = 5 + n + 16 * n
            if len(body) < expected:
                return [], None, None, None, None
            mask_start = 5
            token_start = mask_start + n
            position_start = token_start + 4 * n
            action_start = position_start + 4 * n
            action_logp_start = action_start + 4 * n
            mask = [bool(x) for x in body[mask_start:token_start]]
            tokens = list(struct.unpack(f"!{n}i", body[token_start:position_start])) if n else []
            positions = list(struct.unpack(f"!{n}i", body[position_start:action_start])) if n else []
            actions = list(struct.unpack(f"!{n}i", body[action_start:action_logp_start])) if n else []
            action_logps = list(struct.unpack(f"!{n}f", body[action_logp_start:expected])) if n else []
            return mask, tokens, positions, actions, action_logps

        # OPD2 payload from opd.patches.vllm.speculative_decode:
        #   b"OPD2" + uint8(n) + n mask bytes
        #   + n big-endian int32 token ids + n big-endian int32 response positions.
        if body.startswith(b"OPD2") and len(body) >= 5:
            n = body[4]
            expected = 5 + n + 8 * n
            if len(body) < expected:
                return [], None, None, None, None
            mask_start = 5
            token_start = mask_start + n
            position_start = token_start + 4 * n
            mask = [bool(x) for x in body[mask_start:token_start]]
            tokens = list(struct.unpack(f"!{n}i", body[token_start:position_start])) if n else []
            positions = list(struct.unpack(f"!{n}i", body[position_start:expected])) if n else []
            return mask, tokens, positions, None, None

        logger.error("[opd-rollout] unsupported mask IPC payload prefix: %r", body[:4])
        return [], None, None, None, None

    @staticmethod
    def _align_mask_by_positions(
        mask: list[bool],
        event_tokens: list[int] | None,
        event_positions: list[int],
        token_ids: list[int] | None,
        request_id: str,
    ) -> list[bool]:
        if token_ids is None:
            raise RuntimeError("OPD2 position alignment requires final token_ids")
        expected_len = len(token_ids)
        aligned = [False] * expected_len
        if not (len(mask) == len(event_positions) and (event_tokens is None or len(event_tokens) == len(mask))):
            raise RuntimeError(
                f"OPD2 malformed event lengths for request_id={request_id}: "
                f"mask={len(mask)}, tokens={None if event_tokens is None else len(event_tokens)}, "
                f"positions={len(event_positions)}, expected={expected_len}"
            )
        seen_positions: set[int] = set()
        ignored_oob = 0
        for i, (is_teacher, pos) in enumerate(zip(mask, event_positions)):
            pos = int(pos)
            if pos < 0:
                raise RuntimeError(
                    f"OPD2 position out of range for request_id={request_id}: "
                    f"pos={pos}, expected_len={expected_len}, event_index={i}"
                )
            if pos >= expected_len:
                # vLLM can sample a token that is later excluded from the final
                # RequestOutput (for example a terminal/stop-side event). The
                # absolute coordinate makes this deterministic: events outside
                # final token_ids are not part of the training response mask.
                ignored_oob += 1
                continue
            if event_tokens is not None:
                event_tok = int(event_tokens[i])
                final_tok = int(token_ids[pos])
                if event_tok != final_tok:
                    raise RuntimeError(
                        f"OPD2 token mismatch for request_id={request_id}: "
                        f"pos={pos}, event_token={event_tok}, final_token={final_tok}, event_index={i}"
                    )
            aligned[pos] = aligned[pos] or bool(is_teacher)
            seen_positions.add(pos)
        missing_positions = sorted(set(range(expected_len)) - seen_positions)
        if missing_positions:
            raise RuntimeError(
                f"OPD2 position coverage incomplete for request_id={request_id}: "
                f"covered={len(seen_positions)}, expected={expected_len}, "
                f"missing_head={missing_positions[:8]}, ignored_oob={ignored_oob}"
            )
        return aligned

    @staticmethod
    def _align_actions_by_positions(
        event_positions: list[int],
        action_tokens: list[int] | None,
        action_logps: list[float] | None,
        token_ids: list[int],
    ) -> tuple[list[int] | None, list[float] | None, list[bool] | None]:
        if action_tokens is None or action_logps is None:
            return None, None, None
        expected_len = len(token_ids)
        if len(action_tokens) != len(event_positions) or len(action_logps) != len(event_positions):
            raise RuntimeError(
                "OPD3 malformed action lengths: "
                f"actions={len(action_tokens)}, action_logps={len(action_logps)}, positions={len(event_positions)}"
            )
        aligned_actions = [int(tok) for tok in token_ids]
        aligned_logps = [float("nan")] * expected_len
        aligned_mask = [False] * expected_len
        for action, action_logp, pos in zip(action_tokens, action_logps, event_positions):
            pos = int(pos)
            if pos < 0 or pos >= expected_len:
                continue
            if not math.isfinite(float(action_logp)):
                continue
            aligned_actions[pos] = int(action)
            aligned_logps[pos] = float(action_logp)
            aligned_mask[pos] = True
        return aligned_actions, aligned_logps, aligned_mask

    def pop_with_actions(
        self,
        request_id: str,
        expected_len: int,
        token_ids: list[int] | None = None,
        timeout_s: float = 0.5,
    ) -> tuple[list[bool] | None, list[int] | None, list[float] | None, list[bool] | None]:
        if request_id is None:
            return None, None, None, None
        if token_ids is None or expected_len != len(token_ids):
            raise RuntimeError(
                f"OPD absolute-position alignment requires final token_ids: "
                f"request_id={request_id}, expected_len={expected_len}, "
                f"token_ids_len={None if token_ids is None else len(token_ids)}"
            )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                mask = self.mask_store.get(request_id)
                positions = self.position_store.get(request_id)
                if mask is not None and positions is not None and len(positions) >= expected_len:
                    mask = self.mask_store.pop(request_id)
                    tokens = self.token_store.pop(request_id, None)
                    positions = self.position_store.pop(request_id, None)
                    actions = self.action_store.pop(request_id, None)
                    action_logps = self.action_logp_store.pop(request_id, None)
                    aligned_mask = self._align_mask_by_positions(mask, tokens, positions, token_ids, request_id)
                    aligned_actions, aligned_logps, aligned_action_mask = self._align_actions_by_positions(
                        positions, actions, action_logps, token_ids
                    )
                    return aligned_mask, aligned_actions, aligned_logps, aligned_action_mask
            time.sleep(0.01)
        with self.lock:
            mask = self.mask_store.pop(request_id, None)
            tokens = self.token_store.pop(request_id, None)
            positions = self.position_store.pop(request_id, None)
            actions = self.action_store.pop(request_id, None)
            action_logps = self.action_logp_store.pop(request_id, None)
        if mask is None:
            return None, None, None, None
        if positions is None:
            raise RuntimeError(f"OPD mask IPC payload has no absolute positions for request_id={request_id}")
        aligned_mask = self._align_mask_by_positions(mask, tokens, positions, token_ids, request_id)
        aligned_actions, aligned_logps, aligned_action_mask = self._align_actions_by_positions(
            positions, actions, action_logps, token_ids
        )
        return aligned_mask, aligned_actions, aligned_logps, aligned_action_mask

    def pop_stop_event(self, request_id: str, timeout_s: float = 0.05) -> dict[str, Any] | None:
        if request_id is None:
            return None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                events = self.stop_event_store.pop(request_id, None)
            if events:
                return min(events, key=lambda e: int(e.get("response_pos", 10**18)))
            time.sleep(0.005)
        with self.lock:
            events = self.stop_event_store.pop(request_id, None)
        if not events:
            return None
        return min(events, key=lambda e: int(e.get("response_pos", 10**18)))

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            os.unlink(self.path)
        except Exception:
            pass


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(vLLMHttpServer)

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        self._validate_launch_requirements()

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            env_vars = {
                **{var: "1" for var in get_platform().ray_noset_envvars()},
                **get_platform().rollout_env_vars(),
            }

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                is_reward_model=self.is_reward_model,
                is_teacher_model=self.is_teacher_model,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    async def release_kv_cache(self):
        # Drain all in-flight requests so that vLLM worker threads go idle
        # before we touch engine.release_kv_cache()
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _validate_launch_requirements(self) -> None:
        """Validate requirements before launching. Override in subclasses."""
        # NOTE: We always use MP Executor backend whether it's single-node or multi-node.
        # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
        if self.config.data_parallel_size == 1 and self.nnodes > 1:
            assert _VLLM_VERSION >= version.parse("0.11.1"), (
                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
            )

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_')."""
        return "vllm_"
