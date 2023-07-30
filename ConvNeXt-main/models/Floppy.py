import torch
from ptflops import get_model_complexity_info
from convnext import convnext_tiny
from PIB_Convnext import convnext_tiny as convnext_tiny_pib
from Complex_PIB_Convnext import convnext_tiny as convnext_tiny_pib_complex



import argparse



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--arch', type=str, default='DARTS_PseudoInvBn', help='which architecture to use')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
args = parser.parse_args()

with torch.cuda.device(0):
    # genotype = eval("genotypes.%s" % args.arch)
    # model = NetworkCIFAR(args.init_channels, 10, args.layers, args.auxiliary, genotype)
    model = convnext_tiny_pib_complex()


    net1 = model.to('cuda:0')
    macs, params = get_model_complexity_info(net1, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # net2 = MobileNetV2().to('cuda:0')
    # macs2, params2 = get_model_complexity_info(net2, (3, 32, 32), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs2))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params2))