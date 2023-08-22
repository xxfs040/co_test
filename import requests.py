import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

def get_data(start_date, end_date):
    url = "http://www.sz-mtr.com/service/information/data/index.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="tableList")
    rows = table.find_all("tr")
    data = []
    for row in rows[1:]:
        cols = row.find_all("td")
        date = datetime.strptime(cols[0].text, "%Y-%m-%d")
        if start_date <= date <= end_date:
            daily_flow = int(cols[10].text.replace(",", ""))
            data.append((date, daily_flow))
    return data

def main():
    start_date = datetime(2020, 6, 1)
    end_date = datetime(2022, 12, 31)
    data = get_data(start_date, end_date)
    df = pd.DataFrame(data, columns=["date", "daily_flow"])
    df.to_csv("suzhou_metro_one_line_daily_flow.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
