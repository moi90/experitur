import importlib
import operator
import os
import sys

import click

from experitur import __version__
from experitur.core.context import Context
from experitur.core.experiment import (
    CommandNotFoundError,
    TrialNotFoundError,
)
from experitur.core.trial import Trial
from experitur.dox import DOXError, load_dox


@click.group()
@click.version_option(version=__version__)
def cli():  # pragma: no cover
    pass


@cli.command()
@click.argument("dox_fn")
@click.option(
    "--skip-existing/--no-skip-existing", default=True, help="Skip existing trials."
)
@click.option("--catch/--no-catch", default=True, help="Catch trial exceptions.")
@click.option(
    "--clean-failed/--no-clean-failed",
    default=False,
    help="Delete failed trials before running.",
)
@click.option("--yes", "-y", is_flag=True, help="Delete without asking.")
@click.option("--reload", "-r", is_flag=True, help="Reload DOX file if modified.")
def run(dox_fn, skip_existing, catch, clean_failed, yes, reload):
    """Run experiments."""
    click.echo("Running {}...".format(dox_fn))

    # Record DOX modification time
    dox_mtime = os.path.getmtime(dox_fn)

    # Record currently loaded modules
    initial_modules = set(sys.modules.values())

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    config = {
        "skip_existing": skip_existing,
        "catch_exceptions": catch,
    }

    while True:
        with Context(wdir, config, writable=True) as ctx:
            ctx: Context
            if clean_failed:
                selected = [trial for trial in ctx.get_trials() if trial.is_failed]
                if selected:
                    click.echo(
                        "The following {} trials will be deleted:".format(len(selected))
                    )
                    list_trials(selected)
                    if yes or click.confirm("Continue?"):
                        for trial in selected:
                            trial.remove()

            # Load the DOX
            try:
                load_dox(dox_fn)
            except DOXError as exc:
                raise exc.__cause__ from None

            # Run
            ctx.run()

        # TODO: Reload only first-party
        new_dox_mtime = os.path.getmtime(dox_fn)
        if reload and new_dox_mtime > dox_mtime:
            dox_mtime = new_dox_mtime
            # Reload modules and redo loop
            print(f"{dox_fn} was changed, reloading...")
            for module in set(sys.modules.values()) - initial_modules:
                try:
                    importlib.reload(module)
                except:  # pytlint: disable=bare-except
                    print(f"Error reloading {module}")
            continue

        # Exit from loop
        break


@cli.command()
@click.argument("dox_fn")
@click.argument("--clear/--no-clear", default=False)
def stop(dox_fn, clear):
    """Stop experiments after the current trial finished."""
    wdir = os.path.splitext(dox_fn)[0]

    with Context(wdir, writable=True) as ctx:
        if clear:
            ctx.stop(False)
            print("Stop signal cleared.")
        else:
            ctx.stop()
            print("Stop signal set.")


@cli.command(context_settings=dict(ignore_unknown_options=True,))
@click.argument("dox_fn")
@click.argument("cmd")
@click.argument("target")
@click.argument("cmd_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def do(click_ctx: click.Context, dox_fn, target, cmd, cmd_args):
    """Execute experiment subcommands."""
    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir) as ctx:
        # Load the DOX
        load_dox(dox_fn)

        # Run
        try:
            ctx.do(target, cmd, cmd_args)
        except CommandNotFoundError:
            click_ctx.fail("Command not found: {}".format(cmd))
        except TrialNotFoundError:
            click_ctx.fail("Trial not found: {}".format(target))


@cli.command()
@click.argument("dox_fn")
@click.argument("experiment_id", default=None, required=False)
@click.option("--all", is_flag=True, help="Delete all trials.")
@click.option("--yes", "-y", is_flag=True, help="Delete without asking.")
def clean(dox_fn, experiment_id, all, yes):
    """
    Clean trials.

    By default, only failed trials are deleted. Use --all to delete all trials.
    """
    click.echo("Cleaning results from {}...".format(dox_fn))

    wdir = os.path.splitext(dox_fn)[0]
    with Context(wdir) as ctx:
        selected = [
            trial
            for trial in ctx.get_trials(experiment=experiment_id)
            if (trial.is_failed or all)
        ]

        n_selected = len(selected)

        if not n_selected:
            click.echo("No matching trials.")
            if not all:
                click.echo("Use --all to delete all trials.")
            return

        click.echo("The following {} trials will be deleted:".format(n_selected))
        list_trials(selected)

        if yes or click.confirm("Continue?"):
            ctx.store.delete_all(selected.keys())


def _status(trial: Trial):
    if trial.is_failed:
        return "!"

    if getattr(trial, "success", False):
        return " "

    return ">"


def list_trials(trials):
    """Show a sorted list of trials with a status signifier.

    ! Failed
    > Running

    """
    for trial in sorted(trials, key=operator.attrgetter("id")):
        print("{} {}".format(_status(trial), trial.id))


@cli.command()
@click.argument("dox_fn")
@click.argument("experiment_id", default=None, required=False)
def show(dox_fn, experiment_id=None):
    """List trials."""

    wdir = os.path.splitext(dox_fn)[0]
    with Context(wdir) as ctx:
        trials = ctx.get_trials(experiment=experiment_id)
        list_trials(trials)


@cli.command()
@click.argument("dox_fn")
@click.argument("results_fn", required=False, default=None)
@click.option("--failed/--no-failed", default=False, help="Include failed trials.")
def collect(dox_fn, results_fn, failed):
    """Collect results."""
    wdir = os.path.splitext(dox_fn)[0]

    if results_fn is None:
        results_fn = "{}.csv".format(wdir)

    click.echo("Collecting results from {} into {}...".format(dox_fn, results_fn))

    with Context(wdir) as ctx:
        load_dox(dox_fn)

        ctx.collect(results_fn, failed=failed)
