import os
from datetime import datetime
import time


class OutPutDir():
    def __init__(self, _output_dir, _comment=""):
        self.out_dir = _output_dir
        self.comment = _comment

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.normpath(os.path.join(cur_dir, "../"))

        self.now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.parent_dir = os.path.join(self.root_dir,
                                       self.out_dir,
                                       self.now + "_" + self.comment,
                                       )
        self.log_dir = os.path.join(self.parent_dir, "logs")
        self.checkpoint_dir = os.path.join(self.parent_dir, "checkpoint")
        self.result_dir = os.path.join(self.parent_dir, "results")

        self.create_working_dir(self.parent_dir)
        self.create_working_dir(self.log_dir)
        self.create_working_dir(self.checkpoint_dir)
        self.create_working_dir(self.result_dir)

    def create_working_dir(self, dir_name):
        """Create folder under given path.

        Args:
            dir_name (str): Directory path to be created
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
    def get_abs_path(self, path):
        return os.path.join(self.root_dir, path)

    def get_checkpoint_file(self, epoch, comment):
        return os.path.join(self.checkpoint_dir,
                            "model_" + str(epoch) + comment + ".pth")


if __name__ == "__main__":
    o = OutPutDir("output", "test")
    print(o.out_dir)
    print(o.comment)
    print(o.root_dir)
    print(o.now)
    print(o.parent_dir)
    print(o.log_dir)
    print(o.checkpoint_dir)
