import time
import os
from functools import wraps
import types

def print_param(name_output = "error.txt", BASE_DIR = os.path.dirname(os.path.abspath(__file__))):
    """
    print and save result
    :param name_output: .txt file
    :param BASE_DIR: Where to save file
    :return:
    """
    def print_result(func):
        '''
        print_result(sockMerchant(n,ar),"output_sock.txt", BASE_DIR)
        :param func:
        :param name_output:
        :param BASE_DIR:
        :return:
        '''
        @wraps(func)
        def f(*arg, **kwarg):
            output = os.path.join(BASE_DIR, name_output)
            if os.path.exists(output):
                f = open(output, "r+")
            else:
                f = open(output, "w")
            result = func(*arg, **kwarg)
            if isinstance(result, types.GeneratorType):
                f.write('\n'.join(map(str, result)))
                f.write('\n')
            else:
                f.write(str(result))
            f.close()
            myfile = open(output, 'r+')
            print(myfile.readlines())
            return result
        return f
    return print_result

def timer(func):
    @wraps(func)
    def f(*arg, **kwarg):
        before = time.time()
        rv = func(*arg, **kwarg)
        after = time.time()
        print("--- %s seconds ---" %(after - before))
        return rv
    return f


@timer
def add(a,b):
    result = a + b
    return result

if __name__ == "__main__":
    print(add(2,3))
