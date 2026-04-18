import time
from rich.console import Console
from rich.status import Status
import justin.cli as cli

console = Console()
sweeper = cli.SweepingText("JUSTIN is thinking...")
status = console.status(sweeper, spinner="point", spinner_style="#ffb000")

with status:
    time.sleep(2)
    sweeper.update_text("Recalling memories...")
    time.sleep(2)
    sweeper.update_text("Using tool: search_web...")
    time.sleep(2)
