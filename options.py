import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BERT4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='Benchmarks for session-based rec.')
    parser.add_argument('--max_len', '-l', type=int, default=50, help='Length of sequence')
    parser.add_argument('--config_file', '-c', type=str, default='config.yaml', help='Config file path')

    # parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    # parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    return parser.parse_known_args()[0]