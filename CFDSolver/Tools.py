class IncompatibleListException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def calc_error(real: list, calculated: list) -> float:
    if len(real) != len(calculated):
        raise IncompatibleListException
    cum_tot = 0
    for i in range(0, len(real) - 1):
        cum_tot = cum_tot + (calculated[i] - real[i]) / real[i]
    return 100 / len(real) * cum_tot
