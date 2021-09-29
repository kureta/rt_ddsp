"""Console script for rt_ddsp."""
import argparse
import sys


def main() -> int:
    """Console script for rt_ddsp."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "rt_ddsp.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
