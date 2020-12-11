'''Fix polygon geometries in a table by applying a zero-width buffer.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('table', help='table name to fix geometries in')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.PolygonGeometryFixer.fromArgs(args).run(
        args.table,
    )
