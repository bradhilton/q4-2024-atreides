import torch
from vllm import LLM
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.executor.executor_base import ExecutorBase
from typing import Optional, Union


class Tokenizer:
    def __init__(self, model: str) -> None:
        self.llm = get_llm(model)

    def get_token_id(self, token: str) -> int:
        return self.llm.get_tokenizer().convert_tokens_to_ids(token)

    def get_token(self, token_id: int) -> str:
        return self.llm.get_tokenizer().convert_ids_to_tokens(token_id)

    def encode(
        self,
        messages: Union[
            list[ChatCompletionMessageParam], list[list[ChatCompletionMessageParam]]
        ],
        add_generation_prompt: bool = False,
        continue_final_message: bool = True,
        concatenate: bool = False,
        seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        tokenizer = self.llm.get_tokenizer()
        generate = self.llm.generate

        def patch(
            prompts: list[dict[str, str]], *args: object, **kwargs: object
        ) -> list[list[int]]:
            return [
                tokenizer.encode(
                    prompt["prompt"].removesuffix("\n<|im_start|>assistant\n")
                )
                for prompt in prompts
            ]

        self.llm.generate = patch  # type: ignore
        token_ids: list[list[int]] = self.llm.chat(
            messages,  # type: ignore
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        self.llm.generate = generate  # type: ignore
        pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
        if type(messages[0]) != list:
            return torch.tensor(token_ids[0])
        elif concatenate:
            cat_token_ids = [token_id for seq in token_ids for token_id in seq]
            if seqlen is not None:
                cat_token_ids = cat_token_ids[:seqlen]
                cat_token_ids += [pad_id] * (seqlen - len(cat_token_ids))
            return torch.tensor(cat_token_ids)
        else:
            seqlen = (
                seqlen if seqlen is not None else max(len(seq) for seq in token_ids)
            )
            trimmed_ids = [seq[:seqlen] for seq in token_ids]
            padded_ids = [seq + [pad_id] * (seqlen - len(seq)) for seq in trimmed_ids]
            return torch.tensor(padded_ids)

    def decode(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
        if len(token_ids.shape) == 1:
            return self.llm.get_tokenizer().decode(token_ids)  # type: ignore
        return [self.llm.get_tokenizer().decode(token_ids) for token_ids in token_ids]  # type: ignore


def get_llm(model: str) -> LLM:
    _get_executor_cls = LLMEngine._get_executor_cls

    def _get_executor_cls_noop(*args, **kwargs):
        return NoOpExecutor

    LLMEngine._get_executor_cls = _get_executor_cls_noop

    llm = LLM(model=model)

    LLMEngine._get_executor_cls = _get_executor_cls

    return llm


class NoOpExecutor(ExecutorBase):
    uses_ray = False

    def _init_executor(self) -> None:
        pass

    def determine_num_available_blocks(self):
        return (0, 0)

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        pass

    def execute_model(self, execute_model_req):
        return None

    def stop_remote_worker_execution_loop(self):
        return

    def add_lora(self, lora_request):
        return False

    def remove_lora(self, lora_id):
        return False

    def pin_lora(self, lora_id):
        return False

    def list_loras(self):
        return set()

    def add_prompt_adapter(self, prompt_adapter_request):
        return False

    def remove_prompt_adapter(self, prompt_adapter_id):
        return False

    def pin_prompt_adapter(self, prompt_adapter_id):
        return False

    def list_prompt_adapters(self):
        return set()

    def check_health(self):
        pass
