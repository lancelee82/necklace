""""""

"""
import socket
print(socket.gethostname())

import socket
print(socket.gethostbyaddr(socket.gethostname())[0])

import platform
print(platform.node())

import os, platform
if platform.system() == "Windows":
    print(platform.uname().node)
else:
    print(os.uname()[1])   # doesnt work on windows
"""


import socket


def get_hostname(*args, **kwargs):
    hostname = socket.gethostname()
    return hostname


import psutil


def get_nic_ip_map(*args, **kwargs):
    nic2ip = {}
    ip2nic = {}

    info = psutil.net_if_addrs()
    for k, v in info.items():
        for item in v:
            if item[0] == 2 and item[1] != '127.0.0.1':  # ipv4
                ipv4 = item[1]
                nic2ip[k] = ipv4
                ip2nic[ipv4] = k
                break

    return nic2ip, ip2nic


def get_ip_by_nic(nic, *args, **kwargs):
    nic2ip, _ = get_nic_ip_map()
    ipv4 = nic2ip.get(nic)
    return ipv4


def get_ip_by_prefix(pre, *args, **kwargs):
    nic2ip, _ = get_nic_ip_map()
    ipv4 = None
    for nic, ip in nic2ip.items():
        if ip.startswith(pre):
            ipv4 = ip
            break
    return ipv4


def get_ip_by_first(*args, **kwargs):
    nic2ip, _ = get_nic_ip_map()
    ipv4 = None
    for nic, ip in nic2ip.items():
        ipv4 = ip
        break
    return ipv4
