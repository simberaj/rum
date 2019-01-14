import argparse

import rum.model

DESCRIPTION = '''Introspects a trained model for its feature information.'''

argparser = argparse.ArgumentParser(DESCRIPTION)
argparser.add_argument('model_file', help='pickled model file')
argparser.add_argument('-g', '--grouped', action='store_true',
    help=('display information for feature groups as specified by common'
    'naming schema'),
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelIntrospector().run(args.model_file, args.grouped)