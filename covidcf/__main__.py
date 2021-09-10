import click

from .data.cwz import process_cwz
from .evaluation.evaluation import hyperopt_evaluate_command
from .data import process_ictcf, process_rumc
from .evaluation import evaluate_command
from .evaluation.hyperopt import hyperopt_command
from .evaluation.experiment import run_experiments_command


@click.group()
def cli():
    pass


cli.add_command(process_ictcf)
cli.add_command(process_rumc)
cli.add_command(process_cwz)
cli.add_command(evaluate_command)
cli.add_command(hyperopt_command)
cli.add_command(hyperopt_evaluate_command)
cli.add_command(run_experiments_command)
# cli.add_command(hyperopt_eval_command)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cli()
