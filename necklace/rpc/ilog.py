from __future__ import print_function


IL = [
    # default
    1, 2, 3, 4, 5, 6, 7, 8, 9,

    # tndsnc
    # 500000,  # worker service
    # 500001,  # worker server
    # 500002,  # scheduler server
    # 500003,  # scheduler service

]


def il_add(i):
    if i not in IL:
        IL.append(i)


def il_del(i):
    if i in IL:
        IL.remove(i)


def debug(i, *args, **kwargs):
    if i not in IL:
        return

    print(*args, **kwargs)

