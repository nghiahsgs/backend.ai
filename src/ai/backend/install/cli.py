from typing import cast

import click
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import ContentSwitcher, Footer, Header, Label, ListItem, ListView, Static

from ai.backend.install import __version__
from ai.backend.plugin.entrypoint import find_build_root

from .types import InstallModes


class ModeMenu(Static):
    """A ListView to choose InstallModes and a description pane underneath."""

    def __init__(
        self,
        mode: InstallModes | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        if mode is None:
            try:
                find_build_root()
                mode = InstallModes.DEV
            except ValueError:
                mode = InstallModes.PACKAGE
        assert mode is not None
        self._mode = mode

    def compose(self) -> ComposeResult:
        yield Label("The installation mode:\n(up/down to change, enter to select)")
        mode_desc = {
            InstallModes.DEV: "Install for development using the current source checkout",
            InstallModes.PACKAGE: "Install using release packages",
        }
        with ListView(
            id="mode-list", initial_index=list(InstallModes).index(InstallModes(self._mode))
        ):
            for mode in InstallModes:
                yield ListItem(
                    Horizontal(
                        Label(mode, classes="mode-item-title"),
                        Label(mode_desc[mode], classes="mode-item-desc"),
                    ),
                    id=f"mode-{mode.value.lower()}",
                )
        yield Label(id="mode-desc")

    @on(ListView.Selected, "#mode-list", item="#mode-dev")
    def start_dev_mode(self) -> None:
        self.app.sub_title = "Development Setup"
        switcher: ContentSwitcher = cast(ContentSwitcher, self.app.query_one("#top"))
        switcher.current = "dev-setup"

    @on(ListView.Selected, "#mode-list", item="#mode-package")
    def start_package_mode(self) -> None:
        self.app.sub_title = "Package Setup"
        switcher: ContentSwitcher = cast(ContentSwitcher, self.app.query_one("#top"))
        switcher.current = "pkg-setup"


class DevSetup(Static):
    pass


class PackageSetup(Static):
    pass


class InstallerApp(App):
    BINDINGS = [
        ("q", "quit", "Quit the installer"),
    ]
    CSS_PATH = "app.tcss"

    def __init__(self, mode: InstallModes | None = None) -> None:
        super().__init__()
        self._mode = mode

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ContentSwitcher(id="top", initial="mode-menu"):
            yield ModeMenu(self._mode, id="mode-menu")
            with DevSetup(id="dev-setup"):
                yield Label("Development Setup", classes="mode-title")
            with PackageSetup(id="pkg-setup"):
                yield Label("Package Setup", classes="mode-title")
        yield Footer()

    def on_mount(self) -> None:
        header: Header = cast(Header, self.query_one("Header"))
        header.tall = True
        self.title = "Backend.AI Installer"

    def action_quit(self):
        self.exit()


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)
@click.option(
    "--mode",
    type=click.Choice([*InstallModes.__members__], case_sensitive=False),
    default=None,
    help="Override the installation mode. [default: auto-detect]",
)
@click.version_option(version=__version__)
@click.pass_context
def main(
    ctx: click.Context,
    mode: InstallModes | None,
) -> None:
    """The installer"""
    app = InstallerApp(mode)
    app.run()
