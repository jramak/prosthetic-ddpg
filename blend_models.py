import argparse
from pdb import set_trace


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-files', type=str, nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ddpg_agents = []
    for model_name in args.model_files:
        # restore agents from model files and store in ddpg_agents
        print("Restoring from..." + model_name)


if __name__ == "__main__":
    main()
