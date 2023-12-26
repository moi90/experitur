import importlib
import os
import sys
import time
from importlib.machinery import ExtensionFileLoader

import click
import dictdiffer

from experitur import __version__
from experitur.core.context import Context
from experitur.core.experiment import CommandNotFoundError, TrialNotFoundError
from experitur.core.trial import Trial
from experitur.dox import DOXError, load_dox
from experitur.util import cprint


@click.group()
@click.version_option(version=__version__)
def cli():  # pragma: no cover
    pass


class ColoredClickException(click.ClickException):
    def show(self, file=None):
        if file is None:
            file = click.get_text_stream("stderr")
        click.secho("Error: {}".format(self.format_message()), file=file, fg="red")


class UnknownExperimentException(ColoredClickException):
    pass


def _try_get_experiment(ctx: Context, name: str):
    try:
        return ctx.get_experiment(name)
    except KeyError:
        messages = [f"Unknown experiment: {name}"]
        known_experiments = sorted(
            filter(None, (e.name for e in ctx.registered_experiments))
        )
        if known_experiments:
            known_experiments = ", ".join(known_experiments)
            messages.append(f"Known experiments are: {known_experiments}")

        raise UnknownExperimentException("\n".join(messages))


@cli.command()
@click.argument("dox_fn")
@click.argument("experiment_names", nargs=-1)
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
@click.option("--resume/--no-resume", default=False, help="Resume interrupted trials.")
@click.option(
    "-n",
    "n_trials",
    type=int,
    default=None,
    help="Run a maximum of n_trials trials.",
)
@click.option(
    "--capture-locals/--no-capture-locals",
    default=False,
    help="Capture all local variables in tracebacks.",
)
def run(
    dox_fn,
    experiment_names,
    skip_existing,
    catch,
    clean_failed,
    yes,
    reload,
    n_trials,
    resume,
    capture_locals,
):
    """Run experiments."""

    # Record currently loaded modules
    initial_modules = set(sys.modules.values())

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    config = {
        "skip_existing": skip_existing,
        "catch_exceptions": catch,
        "run_n_trials": n_trials,
        "traceback_capture_locals": capture_locals,
        "resume_failed": resume,
    }

    while True:
        # Record starting time
        time_started = time.time()
        with Context(wdir, config, writable=True) as ctx:
            ctx: Context
            if clean_failed:
                cprint(
                    "Detecting failed trials for {}...".format(wdir),
                    color="white",
                    attrs=["dark"],
                )
                selected = ctx.trials.filter(
                    lambda trial: trial.is_failed and not trial.is_resumable
                )
                if selected:
                    click.echo(
                        f"The following {len(selected):,d} trials will be deleted:"
                    )
                    selected.print(descr=False)
                    if yes or click.confirm("Continue?"):
                        for trial in selected:
                            trial.remove()

            cprint("Loading {}...".format(dox_fn), color="white", attrs=["dark"])
            # Load the DOX
            try:
                load_dox(dox_fn)
            except DOXError as exc:
                raise exc.__cause__ from None
            cprint("Loading done.", color="white", attrs=["dark"])

            if experiment_names:
                experiments = [_try_get_experiment(ctx, e) for e in experiment_names]
            else:
                experiments = None

            # Run
            ctx.run(experiments)

        # Reload modules
        dox_mtime = os.path.getmtime(dox_fn)
        if not ctx.should_stop() and reload and dox_mtime > time_started:
            # Reload modules and redo loop
            print(f"{dox_fn} was changed, reloading...")
            for module in set(sys.modules.values()) - initial_modules:
                # Skip builtins
                if not hasattr(module, "__file__"):
                    continue

                # Skip PEP 302-compliant C modules (https://stackoverflow.com/a/39304199/1116842)
                if isinstance(getattr(module, "__loader__", None), ExtensionFileLoader):
                    continue

                # Skip modules without a known file
                if module.__file__ is None:
                    continue

                # Skip modules that were not changed since time_started
                if os.path.getmtime(module.__file__) <= time_started:
                    continue

                try:
                    importlib.reload(module)
                except Exception as exc:  # pylint: disable=bare-except
                    # Ignore errors
                    print(f"Error reloading {module}: {exc}")
            continue

        # Exit from loop
        break


def _format_diff(diff):
    for op, node, values in diff:
        if op == "add":
            updates = ", ".join(f"{k}={v}" for k, v in values)
            print(f"ADD {node}:", updates)
        else:
            raise ValueError(f"Unknown operator: {op}")


@cli.command()
@click.argument("dox_fn")
@click.argument("experiment")
@click.option("--yes", "-y", is_flag=True, help="Update without asking.")
def update(dox_fn, experiment, yes):
    """Update trials by invoking on_update."""

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    config = {}

    with Context(wdir, config, writable=True) as ctx:
        cprint("Loading {}...".format(dox_fn), color="white", attrs=["dark"])
        # Load the DOX
        try:
            load_dox(dox_fn)
        except DOXError as exc:
            raise exc.__cause__ from None
        cprint("Loading done.", color="white", attrs=["dark"])

        _experiment = _try_get_experiment(ctx, experiment)

        updated = []
        for orig_data, trial in _experiment.update():
            diff = list(dictdiffer.diff(orig_data, trial._data))

            if diff:
                print(trial.id)
                updated.append(trial)
                _format_diff(diff)

        if yes or click.confirm("Save updated trials?"):
            for trial in updated:
                trial.save()


@cli.command()
@click.argument("dox_fn")
@click.option("--clear/--no-clear", default=False)
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


@cli.command(context_settings=dict(ignore_unknown_options=True))
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
@click.option("--zombie", is_flag=True, help="Detect zombie trials.")
@click.option(
    "--resumable/--no-resumable", default=False, help="Delete resumable trials."
)
@click.option("--match", "-m", help="Only trials with matching error message or ID.")
@click.option("--yes", "-y", is_flag=True, help="Delete without asking.")
def clean(dox_fn, experiment_id, all, zombie, resumable, yes, match):
    """
    Clean trials.

    By default, only failed trials are deleted. Use --all to delete all trials.
    """
    click.echo("Cleaning results from {}...".format(dox_fn))

    wdir = os.path.splitext(dox_fn)[0]
    with Context(wdir, writable=True) as ctx:
        all_trials = ctx.trials.match(experiment=experiment_id)
        if all:
            selected = all_trials
        else:
            if zombie:
                # Detect and finish zombies
                n_zombies = 0
                for t in all_trials:
                    if t.is_zombie:
                        t.error = "Dead"
                        t.save()
                        n_zombies += 1

                if n_zombies:
                    print(f"Finished {n_zombies} zombies.")

            selected = all_trials.filter(
                lambda trial: (
                    trial.is_failed and (resumable or not trial.is_resumable)
                )
            )

        if match is not None:
            selected = selected.filter(
                lambda trial: (trial.error is not None and match in trial.error)
                or match in trial.id
                or match in trial.experiment.get("meta", {}).get("hostname", "")
            )

        n_selected = len(selected)

        if not n_selected:
            click.echo("No matching trials.")
            if not all:
                click.echo("Use --all to delete all trials.")
            return

        click.echo(f"The following {len(selected):,d} trials will be deleted:")
        selected.sorted().print(descr=False)

        if yes or click.confirm("Continue?"):
            ctx.store.delete_all([t.id for t in selected])


@cli.command()
@click.argument("dox_fn")
@click.argument("experiment_id", default=None, required=False)
@click.option("--failed/--no-failed", default=False, help="Show only failed trials.")
@click.option("--running/--no-running", default=False, help="Show only running trials.")
def show(dox_fn, experiment_id=None, failed=False, running=False):
    """List trials."""

    wdir = os.path.splitext(dox_fn)[0]
    with Context(wdir) as ctx:
        trials = ctx.trials.match(experiment=experiment_id).sorted()

        if failed:
            trials = trials.filter(lambda trial: trial.is_failed)

        if running:
            trials = trials.filter(
                lambda trial: not trial.is_successful and not trial.is_failed
            )

        trials.print()


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


@cli.command(hidden=False)
@click.argument("dox_fn")
@click.argument("fd", type=int)
def private_run_trial(dox_fn, fd):
    print("fd", fd)
    from multiprocessing.connection import Connection

    wdir = os.path.splitext(dox_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Connection(fd) as connection, Context(wdir, writable=True) as ctx:
        load_dox(dox_fn)

        # Notify ready
        connection.send(("ready",))

        cmd, msg = connection.recv()

        if cmd != "run":
            raise ValueError(f"Unexpected message: {(cmd, msg)}")

        trial_data = msg

        experiment_name = trial_data["experiment"]["name"]
        experiment = ctx.get_experiment(experiment_name)

        return experiment.run_trial(...)
