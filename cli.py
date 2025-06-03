from typer import Typer

from commands import tok_train, tokenize_data, utilities
from primer.utilities import get_logger, set_hf_paths

logger = get_logger("cli")

app = Typer()
app.add_typer(utilities.app, name="utils")
app.add_typer(tok_train.app, name="tok")
app.add_typer(tokenize_data.app, name="tok")


if __name__ == "__main__":
    set_hf_paths()
    app()
