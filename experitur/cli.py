import os

import click

from experitur.context import Context, push_context
from experitur.dox import load_dox
from experitur.experiment import Experiment


@click.group()
def cli():  # pragma: no cover
    pass


# TODO: --reload: Rerun the DOX file until all trials are executed
@cli.command()
@click.argument('dox_fn')
@click.option('--skip-existing/--no-skip-existing', default=True)
@click.option('--catch/--no-catch', default=True)
@click.option('--clean-failed/--no-clean-failed', default=False)
@click.option('--yes', '-y', is_flag=True)
def run(dox_fn, skip_existing, catch, clean_failed, yes):
    click.echo('Running {}...'.format(dox_fn))

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    config = {
        "skip_existing": skip_existing,
        "catch_exceptions": catch,
    }

    with push_context(Context(wdir, config)) as ctx:
        if clean_failed:
            selected = {trial_id: trial for trial_id,
                        trial in ctx.store.items() if trial.is_failed}

            click.echo(
                "The following {} trials will be deleted:".format(len(selected)))
            list_trials(selected)
            if yes or click.confirm('Continue?'):
                ctx.store.delete_all(selected.keys())

        load_dox(dox_fn)
        ctx.run()


@cli.command()
@click.argument('dox_fn')
@click.argument('experiment_id', default=None, required=False)
@click.option('--all', is_flag=True)
@click.option('--yes', '-y', is_flag=True)
def clean(dox_fn, experiment_id, all, yes):
    """Clean failed experiments."""
    click.echo('Cleaning results from {}...'.format(dox_fn))

    wdir = os.path.splitext(dox_fn)[0]
    with push_context(Context(wdir)) as ctx:
        selected = {trial_id: trial for trial_id,
                    trial in ctx.store.items() if trial.is_failed or all}

        click.echo(
            "The following {} trials will be deleted:".format(len(selected)))
        list_trials(selected)

        if yes or click.confirm('Continue?'):
            ctx.store.delete_all(selected.keys())


def list_trials(trials):
    """Show a sorted list of trials with a status signifier.

    ! Failed
    """
    for trial_id, trial in sorted(trials.items()):
        status = "!" if trial.is_failed else " "

        print("{} {}".format(status, trial_id))


@cli.command()
@click.argument('dox_fn')
@click.argument('experiment_id', default=None, required=False)
def show(dox_fn, experiment_id=None):
    """List the trials of this DOX."""

    wdir = os.path.splitext(dox_fn)[0]
    with push_context(Context(wdir)) as ctx:
        list_trials(ctx.store)


@cli.command()
@click.argument('dox_fn')
@click.argument('results_fn', required=False, default=None)
@click.option('--failed/--no-failed', default=False)
def collect(dox_fn, results_fn, failed):
    wdir = os.path.splitext(dox_fn)[0]

    if results_fn is None:
        results_fn = "{}.csv".format(wdir)

    click.echo('Collecting results from {} into {}...'.format(
        dox_fn, results_fn))

    with push_context(Context(wdir)) as ctx:
        load_dox(dox_fn)

        ctx.collect(results_fn, failed=failed)
