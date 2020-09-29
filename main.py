import argparse
import sys
sys.path.append("../")

from mosquito_class_1.utils.gen_utils import OutPutDir
from mosquito_class_1.modules.trainer import Trainer
import mosquito_class_1.utils.yaml_parser as yaml_parser

# python main.py --settings vgg16


def arg_parser():
    """CLI arg parser.

    Returns:
        [dict]: CLI args
    """
    parser = argparse.ArgumentParser(description='Proxemo Runner')
    parser.add_argument('--settings', type=str, default='vgg11', metavar='s',
                        help='config file for running the network.')
    cli_args = parser.parse_args()

    args = yaml_parser.load(cli_args.settings)
    args["GENERAL"]["SETTINGS_FILE"] = cli_args.settings

    return args


def main():
    """Main routine."""
    args = arg_parser()

    out_dir = OutPutDir(args["GENERAL"]["OUT_PATH"],
                        args["GENERAL"]["COMMENT"])
    yaml_parser.backup(args["GENERAL"]["SETTINGS_FILE"],
                       out_dir.parent_dir)
    # Build model
    model = Trainer(args, out_dir)

    if args['GENERAL']['MODE'] == 'Train':
        model.train()

    elif args['GENERAL']['MODE'] == 'Test':
        model.test()


if __name__ == '__main__':
    main()
