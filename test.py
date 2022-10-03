import requests

url = "https://www.vivino.com/api/explore/explore?country_code=AU&country_codes[]=pt&currency_code=AUD&grape_filter=varietal&min_rating=1&order_by=price&order=asc&page=1&price_range_max=80&price_range_min=20&wine_type_ids[]=1"

r = requests.get(url)

# Your data:
r.json()