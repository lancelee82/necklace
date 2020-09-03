"""necklace hook base"""


class NklcHookBase(object):
    r"""Necklace Hook Function Base.

    A hook is always called before or after a training event like `epoch_end`,
    please reference :class:`~necklace.trainer.tnopbs.ModelWrapperHooks` for detail.

    .. note::
        A hook may be just a function, but suggests to use a callable class (implemented method __call__),
        in which we can save some useful informations when call it.

    Arguments:
        net: Model Network for training.
        ctx: Context for training, now mainly is the device id.

    .. note::
        Any other arguments could be passed in and saved for use.
    """

    def __init__(self, net, ctx, *args, **kwargs):
        self.net = net
        self.ctx = ctx

    def __call__(self, trnr, *args, **kwargs):
        # NOTE: the first param trnr is just the trainer assigned by `trainer_cls`,
        # and you can use all its public attributes such like `trnr.net`, `trnr.opt`
        raise NotImplemented()
