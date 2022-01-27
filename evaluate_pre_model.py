from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from logging import getLogger
from recbole.utils import init_logger, set_color
from recbole.data import create_dataset, data_preparation
from config import get_config


config = get_config()

# logger initialization
init_logger(config)
logger = getLogger()
logger.info(config)

model_file = 'saved/SASRec-Jan-24-2022_21-35-49.pth'

old_config, model, old_dataset, old_train_data, old_valid_data, old_test_data = load_data_and_model(
    model_file=model_file,
)
logger.info(model)

# dataset filtering
dataset = create_dataset(config)
# print(dataset.ent_feat)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# trainer loading and initialization
trainer = Trainer(config, model)

# model evaluation
test_result = trainer.evaluate(test_data, model_file=model_file)


logger.info(set_color('test result', 'yellow') + f': {test_result}')