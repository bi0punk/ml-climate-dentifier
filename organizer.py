"""
Code browser example.

Run with:

    python code_browser.py PATH
"""

import sys

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import var
from textual.widgets import DirectoryTree, Footer, Header

class CodeBrowser(App):
    """Textual code browser app."""

    CSS_PATH = "code_browser.tcss"
    BINDINGS = [
        ("f", "toggle_files", "Toggle Files"),
        ("q", "quit", "Quit"),
    ]

    show_tree = var(True)

    def watch_show_tree(self, show_tree: bool) -> None:
        """Called when show_tree is modified."""
        self.set_class(show_tree, "-show-tree")

    def compose(self) -> ComposeResult:
        """Compose our UI."""
        path = "./" if len(sys.argv) < 2 else sys.argv[1]
        yield Header()
        with Container():
            yield DirectoryTree(path, id="tree-view")
        yield Footer()

    def action_toggle_files(self) -> None:
        """Called in response to key binding."""
        self.show_tree = not self.show_tree

if __name__ == "__main__":
    CodeBrowser().run()
