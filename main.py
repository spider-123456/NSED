
import argparse
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from data_set_beibei import DataSet
from model_cascade import NSED
from trainer_beibei import Trainer

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':
    # 自定义解析器函数，将字符串转换为列表
    def parse_task_weight(s):
        try:
            # 以逗号分隔字符串，并将每个部分转换为浮点数，组成列表返回
            return [float(x) for x in s.split(',')]
        except ValueError:
            # 解析失败时抛出错误
            raise argparse.ArgumentTypeError('Invalid task weight format. Please use comma-separated float numbers.')

    def gnn_layer(s):
        try:
            # 以逗号分隔字符串，并将每个部分转换为浮点数，组成列表返回
            return [int(x) for x in s.split(',')]
        except ValueError:
            # 解析失败时抛出错误
            raise argparse.ArgumentTypeError('Invalid task weight format. Please use comma-separated float numbers.')

    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--run_ncl_ssl', type=int, default=1, help='(bool) ssl_reg')
    parser.add_argument('--run_ncl_proto', type=int, default=0, help='(bool)')
    parser.add_argument('--run_sgl', type=int, default=1, help='(bool) ssl_weight')
    parser.add_argument('--run_inter', type=int, default=1, help='(bool) inter_reg')

    parser.add_argument('--ssl_reg', type=float, default=2e-5, help='ncl intra_reg')
    parser.add_argument('--proto_reg', type=float, default=5e-4, help='')
    parser.add_argument('--inter_reg', type=float, default=1e-5, help='behavior graph inter loss')
    parser.add_argument('--ssl_weight', type=float, default=1e-5, help='sub_graph ssl_reg')

    parser.add_argument('--temp', type=float, default=0.1, help='(float) The temperature in softmax.')
    parser.add_argument('--ssl_tau', type=float, default=0.5, help='(float) The temperature in softmax.')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layers', type=gnn_layer, default=[2,3,3])
    parser.add_argument('--data_name', type=str, default='tmall', help='')

    parser.add_argument('--hidden_size_list',type=list, default=[64,64,64], help='The hidden size of each layer in GCN layers.')
    parser.add_argument('--node_dropout', type=float, default=0.0,help="The dropout rate of node in each GNN layer.")
    parser.add_argument('--message_dropout', type=float, default=0.1, help='  The dropout rate of edge in each GNN layer.')


    # Contrastive Learning with Structural Neighbors
    # Temperature for contrastive loss.
    # The structure-contrastive loss weight

    # The weight to balance self-supervised loss for users and items.
    parser.add_argument('--structural_alpha', type=float, default=1.0, help='')
    #
    parser.add_argument('--hyper_layers', type=int, default=1, help='')
    parser.add_argument('--gama', type=int, default=1, help='')

    #sgl
    parser.add_argument('--type', type=str, default='ED', help='(str) The type to generate views. Range in ED ND RW')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='(float) The dropout ratio.')
    parser.add_argument('--task_weight', type=parse_task_weight, default=[0.3, 0.3, 0.4], help='(list) The fusion weight.')
    # parser.add_argument('--task_weight', type=list, default=[0.3, 0.3, 0.4], help='(list) The fusion weight.')

    parser.add_argument('--num_clusters', type=int, default=200, help='')
    parser.add_argument('--m_step', type=int, default=1, help='')
    parser.add_argument('--warm_up_step', type=int, default=20, help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg',"recall",'mrr'], help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    # parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=3072, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=600, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')

    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multi_head_attention')
    parser.add_argument('--att_dim', type=int, default=64, help='')
    args = parser.parse_args()
    if args.data_name == 'tmall':
        #user 41738 11953
        args.data_path = './data/Tmall'

        # args.behaviors = ['collect', 'cart', 'buy']
        # args.behaviors = ['click', 'cart', 'buy']
        # args.behaviors = ['click', 'collect', 'buy']
        args.behaviors = ['click', 'collect', 'cart','buy']
        # args.layers = [3, 3, 3, 3]
        # args.model_name = 'Tmall'

    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        # args.behaviors = ['buy']
        # args.behaviors = ['cart', 'buy']
        # args.behaviors = ['view', 'buy']
        args.behaviors = ['view', 'cart','buy']
        # args.layers = [1, 1, 1]
        # args.layers = [3, 3, 3]
        args.model_name = 'Beibei'

    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['buy']
        # args.behaviors = ['cart', 'buy']
        # args.behaviors = ['view',  'buy']
        args.behaviors = ['view', 'cart', 'buy']
        # args.layers = [3, 3, 3]
        args.model_name = 'Taobao'
    elif args.data_name == 'tmall1':
        #user 41738 11953
        args.data_path = './data/Tmall1'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [3, 3, 3, 3]
        args.model_name = 'Tmall'


    elif args.data_name == 'beibei1':
        args.data_path = './data/beibei1'
        args.behaviors = ['view', 'cart', 'buy']
        # args.layers = [1, 1, 1]
        args.layers = [3, 3, 3]
        args.model_name = 'beibei1'
    elif args.data_name == 'ml':
        args.data_path = './data/other_data/ML10M'
        args.behaviors = ['neutral', 'neg', 'pos']
        args.layers = [2, 3, 3]
        args.model_name = 'ml'
        args.task_weight = [1,3,3]
    elif args.data_name == 'yelp':
        args.data_path = './data/other_data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
        args.layers = [3, 3, 3, 3]
        args.model_name = 'yelp'
        args.task_weight = [2, 1, 1, 2]


    elif args.data_name == 'taobao2':
        args.data_path = './data/taobao2'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [3, 3, 3]
        args.model_name = 'taobao'

    elif args.data_name == 'ali':
        args.data_path = './data/ali'
        args.behaviors = ['view', 'click', 'conversion']
        args.layers = [3, 3, 3]
        args.model_name = 'ali'

    elif args.data_name == 'retail':
        args.data_path = './data/other_data/Retail'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [2, 3, 3]
        args.model_name = 'retail'
        args.task_weight = [1,3,3]
    # elif args.data_name == 'jdata':
    #     args.data_path = './data/jdata'
    #     args.behaviors = ['view', 'collect', 'cart', 'buy']
    #     args.layers = [1, 1, 1, 1]
    #     args.model_name = 'jdata'
    #     args.task_weight = [0.2, 0.2, 0.3, 0.3]
    # elif args.data_name == 'tmall2':
    #     args.data_path = './data/tmall15449'
    #     args.behaviors = ['view', 'cart', 'buy']
    #     args.layers = [3, 3, 3]
    #     args.model_name = 'tmall15449'
    #     args.task_weight = [0.2, 0.2, 0.3, 0.3]
    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_enb_{}_{}'.format(args.model_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = NSED(args, dataset)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 12, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



