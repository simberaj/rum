'''Introspect a trained model and compute its SHAP values.'''

import argparse

import rum.model

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('model_file', help='pickled model file')
argparser.add_argument('folder', help='folder to save SHAP plot outputs')
# argparser.add_argument('-g', '--grouped', action='store_true',
    # help=('display information for feature groups as specified by common'
    # 'naming schema'),
# )
argparser.add_argument('-S', '--sample', type=int,
    help='only compute SHAP values on this many samples to speed up the process',
)
argparser.add_argument('-s', '--seed', type=int, 
    help='seed for random generator (initializes the sample selection)'
)
argparser.add_argument('-C', '--no-condition', action='store_true',
    help='do not use modeling condition even if present'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.SHAPGetter.fromArgs(args).run(
        args.model_file,
        args.folder,
        subsampleN=args.sample,
        seed=args.seed,
        condition=(not args.no_condition),
    )
