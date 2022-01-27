from recbole.config import Config
from options import get_args
import os

args = get_args()

# configurations initialization
def get_config():
    config_dict = {
        'gpu_id' : '0',
        'epochs' : 100,
        'loss_type' : 'CE',
        'neg_sampling' : 'None',
        'MAX_ITEM_LIST_LENGTH' : f'{args.max_len}',

        'metrics' : ['Recall', 'NDCG'],
        'valid_metric': 'NDCG@10',
        'topk' : [1,5,10,20,50],
        'save_dataset' : True,
        'load_col' : {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'eval_args' : {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'uni100', 'group_by': 'user'}
    }

    if os.path.isfile(f'dataset/{args.dataset}/{args.dataset}.yaml'):
        config = Config(model=args.model, dataset=args.dataset, config_dict=config_dict, config_file_list=[f'dataset/{args.dataset}/{args.dataset}.yaml'])
    else:
        config = Config(model=args.model, dataset=args.dataset, config_dict=config_dict)
    return config