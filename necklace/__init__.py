""""""

# ----------------------------------------------------------
# do some special initiation for necklace

import multiprocessing as mp

def init_mp():
    # NOTE: to avoid fork new process and avoid conflict with torch.multiprocessing
    # import multiprocessing as mp
    # print(mp.get_start_method())  # fork
    # mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    # print(mp.get_start_method())  # spawn


from necklace.cuda import nbutils

def init_nb(gpu_dev_i=None):
    # NOTE: here we should do selecting gpu device firstly before
    # any other import and use of numba, otherwise the error of
    # core dump maybe occur sometimes when using numba.cuda
    # from necklace.cuda import nbutils

    if gpu_dev_i is not None:
        try:
            gpu_dev_i = int(gpu_dev_i)
            nbutils.cuda_select_device(gpu_dev_i)
        except Exception as ex:
            pass


# NOTE: this should be called just after import necklace
def init(gpu_dev_i):

    init_mp()

    init_nb(gpu_dev_i)
