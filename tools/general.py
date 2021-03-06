import os


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def ensure_dir(output_dir: str):
    """
    Checks whether all directories in <output_dir> exist and creates them if not.

    :param output_dir:              desired directory
    """
    if output_dir[0] == '/':
        current_dir = '/'

    else:
        current_dir = ''

    for s in output_dir.split('/'):
        current_dir += s

        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

        current_dir += '/'
