from textual.app import App, ComposeResult
from textual.widgets import Header, Digits

class CombinedApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #pi {
        border: double green;
        width: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Digits("25.0%", id="pi")

    def on_mount(self) -> None:
        self.title = "Header Application"
        self.sub_title = "With title and sub-title"

if __name__ == "__main__":
    app = CombinedApp()
    app.run()
