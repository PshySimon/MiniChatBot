import random
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import json
import os

from ..data_utils import BaseTextReader

random.seed(42)


class PreTrainedTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space


class TokenizerTrainer:
    def __init__(
        self,
        reader: BaseTextReader,
        vocab_size: int,
        output_dir: os.PathLike,
        add_prefix_space: bool = False,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        additional_special_tokens: list = [],
        legacy: bool = False,
        use_default_system_prompt: bool = False,
        chat_template: str = None,
    ):
        self.reader_iterator = reader()
        self.output_dir = output_dir
        self.add_prefix_space = False
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.additional_special_tokens = additional_special_tokens
        self.legacy = legacy
        self.use_default_system_prompt = use_default_system_prompt
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=add_prefix_space
        )
        self.special_tokens = ["<unk>", "<s>", "</s>"]
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.chat_template = chat_template

    def train(self) -> None:
        self.tokenizer.train_from_iterator(self.reader_iterator, trainer=self.trainer)
        self.tokenizer.decoder = decoders.ByteLevel()

        assert self.tokenizer.token_to_id("<unk>") == 0
        assert self.tokenizer.token_to_id("<s>") == 1
        assert self.tokenizer.token_to_id("</s>") == 2

        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(self.output_dir, "tokenizer.json"))
        self.tokenizer.model.save(self.output_dir)

        custom_tokenizer = PreTrainedTokenizer(
            tokenizer_file=os.path.join(self.output_dir, "tokenizer.json"),
            add_prefix_space=self.add_prefix_space,
            add_bos_token=self.add_bos_token,
            add_eos_token=self.add_eos_token,
            additional_special_tokens=self.additional_special_tokens,
            legacy=self.legacy,
            use_default_system_prompt=self.use_default_system_prompt,
            chat_template=self.chat_template,
        )
        custom_tokenizer.save_pretrained(self.output_dir)
