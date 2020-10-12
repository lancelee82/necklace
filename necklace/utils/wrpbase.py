"""necklace function wrapper base"""


class NklcWrapperBase(object):
    r"""Necklace Function Wrapper Base.

    A function wrapper is used when a funcion is too complex
    to pass into necklace framework, and we want to keep the
    complexity outside of the necklace core.
    The main use cases of wrapper for now are calculating loss
    and doing metric.
    """

    def __init__(self, net, ctx, *args, **kwargs):
        self.net = net
        self.ctx = ctx

    def __call__(self, *args, **kwargs):
        raise NotImplemented()
