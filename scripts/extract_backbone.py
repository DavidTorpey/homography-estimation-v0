from argparse import ArgumentParser

import torch


def extract_backbone(args):
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))['ssl_model']

    new_state_dict = {}
    s = 'backbone.'
    for k, v in state_dict.items():
        if s in k:
            new_state_dict[k.replace(s, '').replace('base.', '')] = v

    torch.save(new_state_dict, args.export_path)


def parse_args():
    p = ArgumentParser()
    p.add_argument('--checkpoint_path', type=str, required=True)
    p.add_argument('--export_path', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    extract_backbone(args)


if __name__ == '__main__':
    main()
