import argparse
from logging import getLogger

from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from options import get_args
from config import get_config


if __name__ == '__main__':
    args = get_args()
    
    config = get_config()

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    # print(dataset.ent_feat)
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
        train_data, valid_data, test_data, saved=True, show_progress=config['show_progress']
    )

    logger.info(set_color('test result', 'yellow') + f': {test_result}')