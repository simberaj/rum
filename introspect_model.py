import argparse

import rum.model

DESCRIPTION = '''Introspects a trained model for its feature information.'''

argparser = argparse.ArgumentParser(DESCRIPTION)
argparser.add_argument('model_file', help='pickled model file')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelIntrospector().run(args.model_file)