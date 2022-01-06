import argparse
def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmenter_config_path", type=str, default=None)
    parser.add_argument("--aug_type", type=str, default=None, choices=["random", "contextual", "back_translation"])
    parser.add_argument("--aug_pipeline", action="store_true")