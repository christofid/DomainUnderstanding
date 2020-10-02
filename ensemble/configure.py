import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Data path", required=True)

    parser.add_argument("--predictions_path", type=str, help="Path where the predictions of the Transformers models have been stored", required=True)

    parser.add_argument("--output_path", default=None,
                        type=str, help="Output path")


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = vars(parser.parse_args())

    return args
