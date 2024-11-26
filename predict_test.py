import requests

host = "localhost:9696"
url = f"http://{host}/predict"

laptop = {
  "brand": "hp",
  "color": "silver",
  "condition": "Used",
  "gpu": "intel",
  "processor": "intel core i3 8th generation",
  "processor_speed": "2.30",
  "processor_speed_unit": "GHz",
  "type": "notebook/laptop",
  "display_width": "1366",
  "display_height": "768",
  "os": "windows",
  "storage_type": "ssd",
  "hard_drive_capacity": 512.0,
  "hard_drive_capacity_unit": "gb",
  "ssd_capacity": 512.0,
  "ssd_capacity_unit": "gb",
  "screen_size_inch": "13",
  "ram_size": 16.0,
  "ram_size_unit": "gb"
}

response = requests.post(url, json=laptop).json()
print(response)
