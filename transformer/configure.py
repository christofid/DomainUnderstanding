import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Data path", required=True)

    parser.add_argument("--order_type", type=str, help="Order type", choices=['bfs','random'])

    parser.add_argument("--embedding_path", required=True,
                        type=str, help="Path of pre-trained word embeddings")

    parser.add_argument("--words_path", default='words_dictionary', type=str, help="Path of words dictionary")

    parser.add_argument("--root_node", type=str, help="Root node", default=None)

    parser.add_argument("--output_path", default=None,
                        type=str, help="Output path")

    parser.add_argument("--model_path", default=None,
                        type=str, help="Output model path")

    parser.add_argument("--batch_size", default=128,
                        type=int, help="Batch Size (default: 128)")

    parser.add_argument("--enc_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--enc_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--dec_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--dec_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--enc_pf_dim", default=512,
                        type=int, help="Encoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--dec_pf_dim", default=512,
                        type=int, help="Decoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--enc_dropout", default=0.1,
                        type=int, help="Encoder dropout (default: 0.1)")

    parser.add_argument("--dec_dropout", default=0.1,
                        type=int, help="Decoder dropout (default: 0.1)")

    parser.add_argument("--learning_rate", default=0.0005,
                        type=int, help="Learning rate (default: 0.0005)")

    parser.add_argument("--epochs", default=20,
                        type=int, help="Epochs (default: 20)")

    parser.add_argument("--clip_value", default=1,
                        type=int, help="Clip value (default: 1)")

    parser.add_argument("--seed_value", default=1,
                        type=int, help="Seed value (default: 1)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = vars(parser.parse_args())

    return args



def parse_args_visualizer():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Data path", required=True)

    parser.add_argument("--order_type", type=str, help="Order type", choices=['bfs','random'])

    parser.add_argument("--root_node", type=str, help="Root node", default=None)

    parser.add_argument("--model_path", default=None, required=True,
                        type=str, help="Model path")

    parser.add_argument("--words_path", default='words_dictionary', type=str, help="Path of words dictionary")

    parser.add_argument("--embedding_path", required=True,
                        type=str, help="Path of pre-trained word embeddings")

    parser.add_argument("--enc_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--enc_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--dec_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--dec_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--enc_pf_dim", default=512,
                        type=int, help="Encoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--dec_pf_dim", default=512,
                        type=int, help="Decoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--enc_dropout", default=0.1,
                        type=int, help="Encoder dropout (default: 0.1)")

    parser.add_argument("--dec_dropout", default=0.1,
                        type=int, help="Decoder dropout (default: 0.1)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = vars(parser.parse_args())

    return args


def parse_args_attention_extractor():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Data path", required=True)

    parser.add_argument("--order_type", type=str, help="Order type", choices=['bfs','random'])

    parser.add_argument("--root_node", type=str, help="Root node", default=None)

    parser.add_argument("--model_path", default=None, required=True,
                        type=str, help="Model path")

    parser.add_argument("--words_path", default='words_dictionary', type=str, help="Path of words dictionary")

    parser.add_argument("--embedding_path", required=True,
                        type=str, help="Path of pre-trained word embeddings")

    parser.add_argument("--enc_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--enc_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--dec_layers", default=3,
                        type=int, help="Encoder layers (default: 3)")

    parser.add_argument("--dec_heads", default=10,
                        type=int, help="Encoder heads (default: 10)")

    parser.add_argument("--enc_pf_dim", default=512,
                        type=int, help="Encoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--dec_pf_dim", default=512,
                        type=int, help="Decoder position-wise feedforward dimension (default: 512)")

    parser.add_argument("--enc_dropout", default=0.1,
                        type=int, help="Encoder dropout (default: 0.1)")

    parser.add_argument("--dec_dropout", default=0.1,
                        type=int, help="Decoder dropout (default: 0.1)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = vars(parser.parse_args())

    return args