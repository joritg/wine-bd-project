import requests
import pandas as pd
import json
from collections import defaultdict
import codecs
import os


def get_wine_data(wine_id, year, page):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    }

    # <-- increased the number of reviews to 9999
    api_url = f"https://www.vivino.com/api/wines/{wine_id}/reviews?per_page=50&year={year}&page={page}"

    data = requests.get(
        api_url.format(id=wine_id, year=year, page=page), headers=headers
    ).json()

    return data


r = requests.get(
    "https://www.vivino.com/api/explore/explore",
    params={
        "country_codes[]": ["pt", "fr", "es", "it", "us"],
        "currency_code": "EUR",
        "language": "en",
        "grape_filter": "varietal",
        "min_rating": "1",
        "order_by": "price",
        "order": "asc",
        "page": 40,
        "price_range_max": "60",
        "price_range_min": "45",
        "wine_type_ids[]": "1",
    },
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0"
    },
)

results = [
    (
        t["vintage"]["wine"]["winery"]["name"],
        t["vintage"]["year"],
        t["vintage"]["wine"]["id"],
        f'{t["vintage"]["wine"]["name"]} {t["vintage"]["year"]}',
        t["vintage"]["statistics"]["ratings_average"],
        t["vintage"]["statistics"]["ratings_count"],
        t["prices"][0]["amount"],
    )
    for t in r.json()["explore_vintage"]["matches"]
]
dataframe = pd.DataFrame(
    results,
    columns=["Winery", "Year", "Wine ID",
             "Wine", "Rating", "num_review", "price"],
)

ratings = []
if os.stat("users-new.json").st_size == 0:
    users = defaultdict(dict)
else:
    with open('users-new.json') as json_file:
        users = json.load(json_file)

user = {}

for _, row in dataframe.iterrows():
    page = 1
    while True:
        print(
            f'Getting info about wine {row["Wine ID"]}-{row["Year"]} Page {page}'
        )

        d = get_wine_data(row["Wine ID"], row["Year"], page)

        if not d["reviews"]:
            break

        for r in d["reviews"]:
            # print(r['user'].get('id'))
            # with open('user.json', 'w') as convert_file:
             #   convert_file.write(json.dumps(r['user']))
            user_id = r['user'].get('id')

            if user_id not in users:
                users[user_id] = {}

            user['name'] = r['user'].get('alias')
            user['language'] = r['user'].get('language')
            user[r['id']] = [r['vintage'].get(
                'id'), row['Winery'], row['price'], r['rating']]
            users[r['user'].get('id')].update(user)
            user = {}


            

            '''
            ratings.append(
                [
                    row["Year"],
                    row["Wine ID"],
                    r["rating"],
                    r["note"],
                    r["created_at"],
                ]
            )
            '''

        page += 1
'''
ratings=pd.DataFrame(
    ratings, columns=["Year", "Wine ID", "User Rating", "Note", "CreatedAt"]
)
'''
with codecs.open("users-new.json", "w", encoding="utf-8") as convert_file:
    convert_file.write((json.dumps(users, indent=4)))


'''
df_out=ratings.merge(dataframe)
df_out.sort_values("price")
df_out.to_csv("sek-2000-.csv", index=False)
'''
