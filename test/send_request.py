import argparse
import json
import requests

from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', type=str, help='Choose route: ping, update_index or query')
    parser.add_argument('--data-path', type=str, help='Path to json')
    return parser.parse_args()


args = parse_args()
config = Config()
URL = f'{config.SERVICE_URL}{args.route}'


def main():
    if args.route == 'ping':
        print(requests.get(URL).text)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        response = requests.post(URL, json=json.dumps(data))
        print(response.text)


if __name__ == '__main__':
    main()
