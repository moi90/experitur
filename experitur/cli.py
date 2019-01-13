import click

from experitur.experiment import Experiment


@click.group()
def cli():
    pass


@cli.command()
@click.argument('experiment_file')
def run(experiment_file):
    click.echo('Running {}...'.format(experiment_file))

    experiment = Experiment(experiment_file)

    experiment.run()
