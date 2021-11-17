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

class OutpuArguments(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)
        self.parser.add_argument(
            "--dir-path",
            type=str,
            default=".",
            help="Directory to write output")
        self.parser.add_argument(
            "--output-file",
            type=str,
            default="prueba",
            help="File to write output")
    def _correct(self):
        assert isinstance(self.args.dir_path, str)
        assert isinstance(self.args.output_file, str)

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
    def _correct(self):
        ## Output arguments
        assert isinstance(self.args.dir_path, str)
        assert isinstance(self.args.output_file, str)

        ## Main arguments
        assert isinstance(self.args.d, int)
        assert isinstance(self.args.seed, int)
        assert isinstance(self.args.initial_iter, int)
        assert isinstance(self.args.total_iter, int)
        assert isinstance(self.args.lower_bound, int)
        assert isinstance(self.args.upper_bound, int)
        