import argparse
from src.data_utils import JsonLineReader
from src.tokenizers import TokenizerTrainer


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer from corpus.")

    # 添加参数
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output files."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="vocab_size for tokenizer."
    )

    # 解析参数
    args = parser.parse_args()

    # 打印参数（或进行其他处理）
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    reader = JsonLineReader(args.data_path, "$.text")
    trainer = TokenizerTrainer(
        reader,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        chat_template="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}",
    )
    trainer.train()


if __name__ == "__main__":
    main()