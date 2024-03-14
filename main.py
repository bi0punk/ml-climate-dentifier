from textual.app import App
from textual.widgets import Header, Footer, Button
from textual.reactive import Reactive

class WeatherDataApp(App):
    # Reactive states for temperature and humidity data
    temperature = Reactive("25.0°C")
    humidity = Reactive("40%")

    async def on_mount(self) -> None:
        await self.view.dock(Header(title="Weather Data Dashboard"), edge="top")
        await self.view.dock(Footer(), edge="bottom")

        # Create buttons to simulate updating temperature and humidity data
        self.temp_button = Button(label=f"Temperature: {self.temperature}", name="update_temperature")
        self.hum_button = Button(label=f"Humidity: {self.humidity}", name="update_humidity")

        # Dock buttons in the app's view
        await self.view.dock(self.temp_button, edge="top")
        await self.view.dock(self.hum_button, edge="top")

    async def handle_button_pressed(self, message) -> None:
        # Simulate data update
        if message.sender.name == "update_temperature":
            # Update temperature (this could be from an actual data source)
            self.temperature = "26.0°C"
            self.temp_button.label = f"Temperature: {self.temperature}"
        elif message.sender.name == "update_humidity":
            # Update humidity (this could be from an actual data source)
            self.humidity = "45%"
            self.hum_button.label = f"Humidity: {self.humidity}"

        # Update the UI to reflect the new state
        await self.temp_button.refresh()
        await self.hum_button.refresh()

if __name__ == "__main__":
    app = WeatherDataApp()
    app.run()
