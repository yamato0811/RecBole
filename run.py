from recbole.quick_start import run_recbole

parameter_dict = {
    'epochs' : 50,
    'loss_type' : 'CE',
    'neg_sampling' : 'None',

    'metrics' : ['Recall', 'NDCG', 'MRR'],
    'topk' : [1,5,10,20,50],
    'save_dataset' : True,
    'load_col' : {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
    'eval_args' : {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'uni100', 'group_by': 'user'}
}
run_recbole(model='SASRec', dataset='ml-1m', config_dict=parameter_dict)
