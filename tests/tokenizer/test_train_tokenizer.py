import os
import pytest
from pathlib import Path
from src.tokenizers import TokenizerTrainer
from src.data_utils import JsonLineReader
from transformers import AutoTokenizer


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    global current_dir
    current_dir = Path(__file__).parent
    yield
    # 测试后的操作
    print("Test end ...")


def test_train_tokenizer():
    global current_dir
    reader = JsonLineReader(
        os.path.join(current_dir, "../data/datasets/test_train_tokenizers_data.jsonl"),
        "$.text",
    )
    trainer = TokenizerTrainer(
        reader,
        500,
        output_dir=os.path.join(current_dir, "../data/tokenizer"),
        chat_template="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}",
    )
    trainer.train()


def load_tokenizer():
    global current_dir
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_dir, "../data/tokenizer"))
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    model_inputs = tokenizer(new_prompt)

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids)
    assert(response == new_prompt)

