from justin.cli import run_setup_wizard
from justin.config import AgentConfig

config = AgentConfig.from_env()
# we won't run it interactively, just checking syntax
