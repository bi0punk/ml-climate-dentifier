#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <DHT.h>

#define DHTPIN D4     // Definimos el pin donde está conectado el DHT22
#define DHTTYPE DHT22 // Cambiamos el tipo de sensor a DHT22

const char* ssid = "CAPI";     // Tu SSID (nombre de la red WiFi)
const char* password = "NOAH2016"; // Tu contraseña de WiFi
const char* server = "http://192.168.1.82:5000/update"; // IP y puerto de tu servidor Flask

DHT dht(DHTPIN, DHTTYPE);
WiFiClient client;

void setup() {
  Serial.begin(9600);
  dht.begin();
  WiFi.begin(ssid, password);
  Serial.println("");

  // Esperamos a conectarnos
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Conectado a ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Esperamos unos segundos entre mediciones
  delay(2000);

  // Leemos la humedad relativa y la temperatura
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // Comprobamos si alguna lectura falló y, de ser así, salimos del loop para intentarlo de nuevo.
  if (isnan(h) || isnan(t)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(client, server);
    http.addHeader("Content-Type", "application/json");
    String httpRequestData = "{\"temperature\":\"" + String(t) + "\",\"humidity\":\"" + String(h) + "\"}";
    int httpResponseCode = http.POST(httpRequestData);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      Serial.println(response);
    }
    else {
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
  else {
    Serial.println("Error in WiFi connection");
  }

  // Mostramos los resultados en el Serial Monitor por si acaso
  Serial.print(F("Humidity: "));
  Serial.print(h);
  Serial.print(F("%  Temperature: "));
  Serial.print(t);
  Serial.println(F("°C "));
}

