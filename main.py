import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='Benchmarks for session-based rec.')
    # parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    # parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    return parser.parse_known_args()[0]



if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'gpu_id' : '1',
        'epochs' : 50,
        'loss_type' : 'CE',
        'neg_sampling' : 'None',

        'metrics' : ['Recall', 'NDCG'],
        'valid_metric': 'NDCG@10',
        'topk' : [1,5,10,20,50],
        'save_dataset' : True,
        'load_col' : {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'eval_args' : {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'uni100', 'group_by': 'user'}
    }

    config = Config(model=args.model, dataset=args.dataset, config_dict=config_dict, config_file_list=[f'dataset/{args.dataset}/{args.dataset}.yaml'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config['show_progress']
    )

    logger.info(set_color('test result', 'yellow') + f': {test_result}')