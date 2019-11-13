'''Recategorize a categorical attribute.

Given a column in a defined schema table, creates its copy with values
recategorized using given translation rules.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('table', help='table containing the attribute')
argparser.add_argument('source', help='the attribute column to be recategorized')
argparser.add_argument('target', help='the column to create and put the result in')
argparser.add_argument('translation', help='a JSON file defining the translation')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.Recategorizer.fromArgs(args).run(
        args.table, args.source, args.target, args.translation, args.overwrite
    )