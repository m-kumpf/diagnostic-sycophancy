"""Reproduce the primary and supplementary analyses from the released data."""

from . import analysis, figures, inference
from .supplementary import decomposition, high_consistency, multiplicity, plausibility, resampling, specialty


def main():
    resampling.main()
    analysis.main(["--quiet"])
    inference.main(["--quiet"])
    multiplicity.main()
    specialty.main()
    plausibility.main()
    high_consistency.main()
    decomposition.main()
    figures.main()


if __name__ == "__main__":
    main()
