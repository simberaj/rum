import rum
import rum.util

DESCRIPTION = '''Drops all existing feature tables in the given schema.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureClearer.fromArgs(args).run()