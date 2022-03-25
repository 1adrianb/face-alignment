
def _init():
    """ 初始化 """

    global _global_list
    _global_list = [[],[]]


def append0(value):
    """ 定义一个全局变量 """

    _global_list[0].append(value)

def append1(value):
    """ 定义一个全局变量 """

    _global_list[1].append(value)

def get():
    return _global_list