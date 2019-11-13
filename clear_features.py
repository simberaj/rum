'''Drop all existing feature tables in the given schema.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureClearer.fromArgs(args).run()