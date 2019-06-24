import os

import click

from experitur.context import Context, push_context
from experitur.dox import load_dox
from experitur.experiment import Experiment


@click.group()
def cli():
    pass


# TODO: --reload: Rerun the DOX file until all trials are executed
@cli.command()
@click.argument('dox_fn')
@click.option('--skip-existing/--no-skip-existing', default=True)
@click.option('--catch/--no-catch', default=True)
def run(dox_fn, skip_existing, catch):
    click.echo('Running {}...'.format(dox_fn))

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    config = {
        "skip_existing": skip_existing,
        "catch_exceptions": catch,
    }

    with push_context(Context(wdir, config)) as ctx:
        load_dox(dox_fn)
        ctx.run()


@cli.command()
@click.argument('dox_fn')
@click.argument('experiment_id', default=None, required=False)
@click.option('--failed', is_flag=True)
@click.option('--all', is_flag=True)
@click.option('--dry-run', '-n', is_flag=True)
def clean(dox_fn, experiment_id=None, failed=True, all=False, empty=False, successful=False, dry_run=False):
    click.echo('Cleaning failed results from {}...'.format(dox_fn))

    if all:
        click.confirm('Do you really want to permanently delete all results of {}?'.format(
            dox_fn), abort=True)

    failed = failed or all
    empty = empty or all
    successful = successful or all

    if not any((failed, empty, successful)):
        print("Nothing to do. Did you mean --failed?")
        return

    wdir = os.path.splitext(dox_fn)[0]
    with push_context(Context(wdir)) as ctx:
        load_dox(dox_fn)
        ctx.clean(failed=failed, dry_run=dry_run,
                  experiment_id=experiment_id)


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
