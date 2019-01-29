import click

from experitur.experiment import Experiment


@click.group()
def cli():
    pass


# TODO: --reload: Rerun the experiment file until all trials are executed
@cli.command()
@click.argument('experiment_file')
@click.option('--skip-existing/--no-skip-existing', default=True)
@click.option('--halt/--no-halt', default=True)
def run(experiment_file, skip_existing, halt):
    click.echo('Running {}...'.format(experiment_file))

    experiment = Experiment(experiment_file)

    experiment.run(skip_existing=skip_existing, halt_on_error=halt)


@cli.command()
@click.argument('experiment_file')
@click.argument('experiment_id', default=None, required=False)
@click.option('--failed', is_flag=True)
@click.option('--all', is_flag=True)
@click.option('--dry-run', '-n', is_flag=True)
def clean(experiment_file, experiment_id=None, failed=True, all=False, empty=False, successful=False, dry_run=False):
    click.echo('Cleaning failed results from {}...'.format(experiment_file))

    if all:
        click.confirm('Do you really want to permanently delete all results of {}?'.format(
            experiment_file), abort=True)

    failed = failed or all
    empty = empty or all
    successful = successful or all

    if not any((failed, empty, successful)):
        print("Nothing to do. Did you mean --failed?")
        return

    experiment = Experiment(experiment_file)

    experiment.clean(failed=failed, dry_run=dry_run,
                     experiment_id=experiment_id)


@cli.command()
@click.argument('experiment_file')
@click.option('--failed/--no-failed', default=False)
def collect(experiment_file, failed):
    click.echo('Collecting failed results from {}...'.format(experiment_file))

    experiment = Experiment(experiment_file)

    experiment.collect(failed=failed)
