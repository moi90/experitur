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


@cli.command()
@click.argument('experiment_file')
@click.option('--remove-everything', is_flag=True)
def clean(experiment_file, remove_everything):
    click.echo('Cleaning failed results from {}...'.format(experiment_file))

    if remove_everything:
        click.confirm('Do you really want to permanently delete all results of {}?'.format(
            experiment_file), abort=True)

    experiment = Experiment(experiment_file)

    experiment.clean(remove_everything)
