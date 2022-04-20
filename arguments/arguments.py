"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

import argparse
import os
import __main__ as main

class BaseArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--quiet",
            action='store_true',
            help='Fewer information displayed on screen')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        self._correct()

        if not self.args.quiet:
            print('-' * 10 + ' Arguments ' + '-' * 10)
            print('>>> Script: %s' % (os.path.basename(main.__file__)))
            print_args = vars(self.args)
            for key, val in sorted(print_args.items()):
                print('%s: %s' % (str(key), str(val)))
            print('-' * 30)
        return self.args

    def _correct(self):
        """Assert ranges of params, mistypes..."""
        raise NotImplementedError

    def writeArguments(self, filename):
        header  =  ''
        header += ('-' * 10 + ' Arguments ' + '-' * 10 +"\n")
        header += ('>>> Script: %s' % (os.path.basename(main.__file__))+"\n")
        print_args = vars(self.args)
        for key, val in sorted(print_args.items()):
            header += ('%s: %s' % (str(key), str(val))+"\n")
        header += ('-' * 30+"\n")
        file = open(filename,"a+")
        file.write(header)
        file.close()

        return True
class OutpuArguments(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)
        self.parser.add_argument(
            "--dir-path",
            type=str,
            default="experiments",
            help="Directory to write output")
        self.parser.add_argument(
            "--output-file",
            type=str,
            default="prueba",
            help="File to write output")
        self.parser.add_argument(
            "--save",
            action='store_true',
            help='Save resulst in file')
    def _correct(self):
        assert isinstance(self.args.dir_path, str)
        assert isinstance(self.args.output_file, str)
        assert isinstance(self.args.save, bool)

class MainArguments(OutpuArguments):
    def initialize(self):
        OutpuArguments.initialize(self)

        # self.parser.add_argument(
        #     "--O",
        #     type=int,
        #     default=2,
        #     help="# of objective to optimize")
        # self.parser.add_argument(
        #     "--M",
        #     type=int,
        #     default=2,
        #     help="# of constrains (TBD)")
        self.parser.add_argument(
            "--d",
            type=int,
            default=1,
            help="input dimension")
        self.parser.add_argument(
            "--seed",
            type=int,
            default=10,
            help="Seed to generate quasi random initialization")
        self.parser.add_argument(
            "--initial-iter",
            type=int,
            default=2,
            help="No, i.e., number of previous sampled points")
        self.parser.add_argument(
            "--total-iter",
            type=int,
            default=5,
            help="Total iteration of the main loop in the algorithm")
        self.parser.add_argument(
            "--lower-bound",
            type=int,
            default=-1,
            help="Lower bound for x value")
        self.parser.add_argument(
            "--upper-bound",
            type=int,
            default=1,
            help="Upper bound for x value")
        self.parser.add_argument(
            "--showplots",
            action='store_true',
            help='Show different plots for each acq function')
        self.parser.add_argument(
            "--showparetos",
            action='store_true',
            help='Show pareto evaluation results')
        self.parser.add_argument(
            "--saveparetos",
            action='store_true',
            help='Save pareto evaluation results')
        self.parser.add_argument(
            "--savename",
            type=str,
            default="exp",
            help="Name to save experiment name")
    def _correct(self):
        ## Output arguments
        assert isinstance(self.args.dir_path, str)
        assert isinstance(self.args.output_file, str)
        assert isinstance(self.args.save, bool)

        ## Main arguments
        assert isinstance(self.args.d, int)
        assert isinstance(self.args.seed, int)
        assert isinstance(self.args.initial_iter, int)
        assert isinstance(self.args.total_iter, int)
        assert isinstance(self.args.lower_bound, int)
        assert isinstance(self.args.upper_bound, int)
        assert isinstance(self.args.showplots, bool)
        assert isinstance(self.args.showparetos, bool)
        assert isinstance(self.args.saveparetos, bool)
        assert isinstance(self.args.savename, str)
        