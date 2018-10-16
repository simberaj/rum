import rum
import rum.validate

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('true_table', help='table containing the true values')
argparser.add_argument('model_table', help='table containing the modeled/disaggregated values')
argparser.add_argument('report_path', help='save HTML reports to this directory')
argparser.add_argument('multiple', help='grid size multiples to validate on', nargs='*', type=int)
argparser.add_argument('-t', '--true-field', help='true value field', default='target')
argparser.add_argument('-m', '--model-field', help='model value field', default='value')
argparser.add_argument('-x', '--xoffset', metavar='distance',
    help='grid x-coordinate offset in the extent CRS', type=float, default=None
)
argparser.add_argument('-y', '--yoffset', metavar='distance',
    help='grid y-coordinate offset in the extent CRS', type=float, default=None
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.validate.ModelMultiscaleValidator.fromArgs(args).run(
        args.true_table,
        args.model_table,
        trueField=args.true_field,
        modelField=args.model_field,
        multiples=args.multiple,
        reportPath=args.report_path,
        xoffset=args.xoffset,
        yoffset=args.yoffset,
    )