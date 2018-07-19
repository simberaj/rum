import rum
import rum.util

DESCRIPTION = '''Lists the already computed features in the grid of the given schema.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureLister.fromArgs(args).run()