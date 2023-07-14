"""args utils"""

from necklace.utils import osutils


def attach_nk_args_parser(parser):
    # CLI
    #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # NOTE: args.epochs is used in trainer explicitly, so here we put it here as a default argument
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train (default: 10)')

    # GPU index on local host
    parser.add_argument('--gpu_rank', '-g', default='0', type=str, help='GPU device to train with')

    parser.add_argument('--url', '-u', type=str, default='ipc:///tmp/nk-wkr-1.ipc',
                        help='worker url')
    parser.add_argument('--server-url', '-s', type=str, default='ipc:///tmp/nk-svr-1.ipc',
                        help='server url')
    parser.add_argument('--role', '-r', type=str, default='worker',
                        help='node role')
    parser.add_argument('--world-size', '-w', type=int, default=1,
                        help='workers number (default is 1)')
    parser.add_argument('--rank', '-k', type=int, default=0,
                        help='worker rank (required)')

    parser.add_argument('--local_rank', '-lk', type=int, default=-1,
                        help='worker local rank (not used for now)')

    parser.add_argument('--rpc_ifname', '-ri', type=str, default='',
                        help='rpc ifname')
    parser.add_argument('--rpc_ip_prefix', '-rp', type=str, default='',
                        help='rpc ip prefix : 192.168.11.')
    parser.add_argument('--comm_ifname', '-ci', type=str, default='',
                        help='comm ifname')

    parser.add_argument('--comm_type', '-x', type=str, default='nccl',
                        help='communication type [mpi,nccl]')

    parser.add_argument('--tmode', '-t', type=str, default='',
                        help='trainer mode')
    parser.add_argument('--dp-size', '-dpsz', type=int, default=0,
                        help='data parallel size')
    parser.add_argument('--mp-size', '-mpsz', type=int, default=0,
                        help='model parallel size')
    parser.add_argument('--pp-size', '-ppsz', type=int, default=0,
                        help='pipeline parallel size')
    parser.add_argument('--zr-size', '-zrsz', type=int, default=0,
                        help='ZeRO parallel size')

    #args = parser.parse_args()
    #return args

    return parser


def args_remake(args, **kwargs):
    wsz = args.world_size
    rank = args.rank

    # -------------------------------------------------------------------------
    # TODO: how to get the gpu_dev_i on diff hosts ??? [=================]
    # <1> here we assumpt every host has the same number of nslot (gpus),
    # so we can get gpu_dev_i from 0 to nslot-1
    # <2> or, we need hosts topology by setting like -H host1:n1,host2:n2,
    # and then get my hostname then my nslot, and my rank ???
    args.gpu_dev_i = args.rank % args.nslot

    # -------------------------------------------------------------------------
    # tcp://192.168.58.193:12000
    # tcp://@{MY_IP}:1200@{LOCAL_RANK}

    local_rank = args.rank % args.nslot

    if args.rpc_ifname:
        my_ip = osutils.get_ip_by_nic(args.rpc_ifname)
    elif args.rpc_ip_prefix:
        my_ip = osutils.get_ip_by_prefix(args.rpc_ip_prefix)
    # TODO: get_ip_prefix_from_schd_ip() --> get_ip_by_prefix()
    else:
        my_ip = osutils.get_ip_by_first()

    # NOTE: the zrpc url should be diff by rank [!!!]
    # TODO: or use @{RANK} in args and replace by rank here [===========]
    #args.url = args.url + '-' + str(args.rank)
    #args.url = args.url + str(args.rank)
    args.url = args.url.replace('@{MY_IP}', my_ip)
    args.url = args.url.replace('@{LOCAL_RANK}', str(local_rank))


    return args
