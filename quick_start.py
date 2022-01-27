from recbole.quick_start import run_recbole
import argparse


parameter_dict = {
    'gpu_id' : '0',
    'epochs' : 50,
    'loss_type' : 'CE',
    'neg_sampling' : 'None',

    'metrics' : ['Recall', 'NDCG', 'MRR'],
    'topk' : [1,5,10,20,50],
    'save_dataset' : True,
    'load_col' : {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
    'eval_args' : {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'uni100', 'group_by': 'user'}
}
run_recbole(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict)
