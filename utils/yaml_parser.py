import yaml
import os
import sys
import shutil


def load(file_name, config_base_path="../zoo/config"):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_base_path = os.path.normpath(os.path.join(cur_dir,
                                                     config_base_path))
    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')

    with open(file_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_dict

def backup(file_name, dst_dir, config_base_path="../zoo/config"):
    """Copies YAML file from one folder to another.

    Args:
        file_name (str): source file path
        dest_folder (str): destination path
        config_base_path (str, optional): Source file path. Defaults to '../modeling/config'.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_base_path = os.path.normpath(os.path.join(cur_dir,
                                                     config_base_path))
    file_path = os.path.join(config_base_path,
                             file_name + '.yaml')
    try:
        shutil.copy(file_path, dst_dir)
    except IOError as e:
        print("Unable to copy yaml file. %s" % e)
    except:
        print("Unexpected error while copying yaml file:", sys.exc_info())


if __name__ == "__main__":
    print(load('vgg_16'))