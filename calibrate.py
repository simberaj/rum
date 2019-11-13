'''Calibrate values to given target values minimizing MAE.'''

import rum
import rum.model

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('table',
    help='table with values to calibrate')
argparser.add_argument('id_field', help='unique identifier of row')
argparser.add_argument('raw_field', help='field with values to calibrate')
argparser.add_argument('calib_field', help='field with values to calibrate to')
argparser.add_argument('output_field', help='name of field to output the calibrated values')
argparser.add_argument('-m', '--multiply-only', action='store_true',
    help='do not fit intercept in the linear model')
argparser.add_argument('-t', '--type', default='sum',
    help='calibration model type (lad, ols, sum, default: sum)')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing calibrated field')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.Calibrator.fromArgs(args).run(
        args.table,
        args.id_field,
        args.raw_field,
        args.calib_field,
        args.output_field,
        type=args.type,
        overwrite=args.overwrite,
        fit_intercept=(not args.multiply_only),
    )