"""
A little notes on python path-related lib:
    os.listdir(path) 
        can accept the relative path
    pathlib         
        This module offers classes representing filesystem paths with semantics appropriate for different operating systems.
    open()
        can accept the relative path
"""
import os
import pathlib

def default_train_sum_path():
    project_path = pathlib.Path(__file__).parent.parent.parent
    return project_path / "train_summary"

def default_dataset_path():
    """
    default project/src/tools/utils.py dirctory
    """
    project_path = pathlib.Path(__file__).parent.parent.parent
    return project_path / "dataset" / "dataset_csv"
 
def exam_path(dirctory):
    """check if a path is valid in the system"""
    pass

def get_files_with_extension(directory, extension):
    """
    The directory here can't solve 
    """
    files = []
    for name in os.listdir(directory):
        if name.endswith(extension):
            files.append(f'{directory}/{name}')
    return files

   
def check_dataset():
    """ if the none-default dataset provided is valid"""
    pass


if __name__ == "__main__":
    # files = get_files_with_extension(default_dataset_path(), "batch")
    # print(files[:10])
    print(default_dataset_path())
    print(str(default_train_sum_path()))
