"""
Console output. On a terminal: a live progress bar per epoch and rich-formatted logs. Otherwise (slurm logs, pipes):
plain timestamped lines. Every message also goes, unformatted, to the run's train.log. Only rank 0 prints.
"""
import logging
import sys
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

logger = logging.getLogger('pelican')
# Writes only to train.log; used for plain-text copies of what the terminal shows as panels/tables.
file_logger = logging.getLogger('pelican.file')
file_logger.propagate = False


def setup_logging(log_file=None, main=True, level=logging.INFO):
    console = Console(stderr=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    if main:
        if console.is_terminal:
            handler = RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True, log_time_format='[%X]')
        else:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_PlainFormatter('%(asctime)s %(message)s', '%H:%M:%S'))
        root.addHandler(handler)
        for h in list(file_logger.handlers):
            file_logger.removeHandler(h)
        if log_file:
            fh = logging.FileHandler(log_file, mode='a')
            fh.setFormatter(_PlainFormatter('%(asctime)s %(levelname)-7s %(message)s', '%Y-%m-%d %H:%M:%S'))
            root.addHandler(fh)
            file_logger.addHandler(fh)
    else:
        root.addHandler(logging.NullHandler())
    return console


class _PlainFormatter(logging.Formatter):
    """Strip rich markup like [bold] from messages written to files."""
    def format(self, record):
        from rich.text import Text
        msg = super().format(record)
        try:
            return Text.from_markup(msg).plain
        except Exception:
            return msg


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'


class UI:
    def __init__(self, console, main=True, log_every=0):
        self.console, self.main, self.log_every = console, main, log_every
        self.interactive = main and console.is_terminal

    def header(self, cfg, model, n_params, sizes, device):
        if not self.main:
            return
        rows = [('run', cfg.run_dir), ('model', f'{cfg.model} ({cfg.task}), {n_params:,} parameters'),
                ('symmetry', f"stabilizer {cfg.stabilizer}" + ('' if cfg.canonicalize else ', no canonicalization')
                 + (', IRC safe' if cfg.irc_safe else '')),
                ('encoder', cfg.encoder), ('data', ', '.join(f'{k} {v:,}' for k, v in sizes.items())),
                ('device', f"{device}{' (float64)' if cfg.double else ''}")]
        if self.interactive:
            t = Table.grid(padding=(0, 2))
            for k, v in rows:
                t.add_row(f'[bold]{k}[/]', str(v))
            self.console.print(Panel(t, title='[bold]PELICAN', expand=False))
        log = file_logger if self.interactive else logger
        for k, v in rows:
            log.info(f'{k}: {v}')

    def epoch(self, epoch, num_epoch, steps):
        return _EpochProgress(self, epoch, num_epoch, steps)

    def epoch_summary(self, epoch, num_epoch, train, valid, lr, seconds, best, summary_fn):
        star = ' [bold green]★ best[/]' if best else ''
        logger.info(f'[bold]epoch {epoch}/{num_epoch}[/]  train {summary_fn(train)}  │  '
                    f'[cyan]valid {summary_fn(valid)}[/]  │  lr {lr:.2e}  {_fmt_time(seconds)}{star}')

    def results(self, results, summary_keys):
        """results: {(which, split): metrics}."""
        if not self.main or not results:
            return
        keys = [k for k in summary_keys if any(k in m for m in results.values())]
        t = Table(title='Test results', show_lines=False)
        t.add_column('checkpoint')
        t.add_column('split')
        for k in keys:
            t.add_column(k, justify='right')
        for (which, split), m in results.items():
            t.add_row(which, split, *[_fmt_metric(m.get(k)) for k in keys])
        if self.interactive:
            self.console.print(t)
        log = file_logger if self.interactive else logger
        for (which, split), m in results.items():
            log.info(f'{which} {split}: ' + '  '.join(f'{k} {_fmt_metric(m.get(k))}' for k in keys))


def _fmt_metric(v):
    if v is None:
        return '-'
    if float(v).is_integer() and abs(v) < 1e6:
        return f'{int(v)}'
    if abs(v) >= 100:
        return f'{v:.0f}'
    return f'{v:.4f}'


class _EpochProgress:
    def __init__(self, ui, epoch, num_epoch, steps):
        self.ui, self.epoch, self.num_epoch, self.steps = ui, epoch, num_epoch, steps
        self.progress = None
        self.t0 = time.time()

    def __enter__(self):
        if self.ui.interactive:
            self.progress = Progress(TextColumn(f'[bold]epoch {self.epoch}/{self.num_epoch}'), BarColumn(), MofNCompleteColumn(),
                                     TimeElapsedColumn(), TextColumn('<'), TimeRemainingColumn(), TextColumn('{task.fields[stats]}'),
                                     console=self.ui.console, transient=True)
            self.progress.start()
            self.task = self.progress.add_task('train', total=self.steps, stats='')
        return self

    def update(self, step, stats):
        if self.progress is not None:
            self.progress.update(self.task, completed=step, stats=stats)
        elif self.ui.main and self.ui.log_every > 0 and step % self.ui.log_every == 0:
            logger.info(f'epoch {self.epoch}/{self.num_epoch} step {step}/{self.steps}  {stats}')

    def __exit__(self, *exc):
        if self.progress is not None:
            self.progress.stop()
        return False
