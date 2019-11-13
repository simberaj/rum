'''Merge multiple schema data for model training purposes.

Copies and merges the all_feats and specified target tables into a defined
schema.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('source_schema', nargs='+', help='schemas to merge')
argparser.add_argument('-t', '--target-tables', nargs='+', metavar='TABLE_NAME',
    help='target table names in each schema to merge'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing schema contents'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.TrainingSchemaMerger.fromArgs(args).run(
        args.source_schema,
        args.target_tables,
        overwrite=args.overwrite
    )