import requests

API_KEY = "27c5d0050b12c89a764f5df909f5385a"
LAT = 13.0827
LON = 80.2707

url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
data = requests.get(url).json()

print(data)
