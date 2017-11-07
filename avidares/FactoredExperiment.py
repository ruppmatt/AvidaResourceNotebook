from itertools import product, repeat


class FactoredExperimentIterator:

    def __init__(self, fexpr):
        self._xfacts = [ list(a) for a in
                        [ zip(repeat(k),v)
                        for k,v in fexpr.items()] ]

    def __len__(self):
        return len(self._xfacts)

    def __next__(self):
        for ndx, expr in enumerate(self._xfacts):
            yield ndx, expr
        raise StopIteration



class FactoredExperiment:

    def __init__(self, factors, procs=4):
        self._settings = {}
        self._factors = factors
        self._factor_names = factors.keys()
        self._ready = False
        self._data = None

    def run_experiments(cfg, pbar=None):
        if pbar:
            pbar.start()

        try:
            for settings in self:
                pass
        except StopIteration:
            pass

        if pbar:
            pbar.finish()

        self._ready = True


    def get_data(self):
        if not self._ready:
            raise LookupError('The experiments have not been completed.')
        return self._data


    def __iter__(self):
        return FactoredExperimentIterator(self.factors)


    def __len__(self):
        return len(self.__iter__())




class ResourceFactoredExperiment(FactoredExperiment):

    def __init__(self, factors):
        self.super(factors)


class GradientFactoredExperiment(FactoredExperiment):

    def __init__(self, factors):
        self.super(factors)
