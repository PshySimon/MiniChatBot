import pytest
from pathlib import Path
from src.data_utils.data_reader import SimpleTextReader, JsonLineReader


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    global current_dir
    current_dir = Path(__file__).parent
    yield
    # 测试后的操作
    print("Test end ...")


def test_plain_text_reader():
    global current_dir
    text_reader = SimpleTextReader(
        f"{current_dir}/../data/datasets/test_pretrain_data.txt")
    for text_data in text_reader():
        assert isinstance(text_data, str)


def test_jsonline_reader():
    global current_dir
    jsonl_reader = JsonLineReader(
        f"{current_dir}/../data/datasets/test_pretrain_data.jsonl", "$.name")
    for jsonl_data in jsonl_reader():
        assert isinstance(jsonl_data, str)
