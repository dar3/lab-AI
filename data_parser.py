import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict


def parse_time(time_str):
    """ Konwertuje string z godziną na obiekt datetime.time, obsługując godziny powyżej 24 """
    hours, minutes, seconds = map(int, time_str.split(":"))

    # Jeśli godzina jest >= 24, przekształcamy ją na następny dzień
    if hours >= 24:
        hours -= 24  # Przesuwamy do zakresu 0-23
        next_day = True
    else:
        next_day = False

    time_obj = datetime.strptime(f"{hours:02}:{minutes:02}:{seconds:02}", "%H:%M:%S").time()

    return time_obj, next_day  # Zwracamy godzinę i informację, czy to następny dzień


def load_graph(csv_path):
    """ Wczytuje plik CSV i buduje graf połączeń komunikacyjnych """
    df = pd.read_csv(csv_path, dtype={'line': str})  # Określamy typ danych dla kolumny "line"

    graph = defaultdict(list)

    for _, row in df.iterrows():
        start_stop = row["start_stop"]
        end_stop = row["end_stop"]
        departure_time, dep_next_day = parse_time(row["departure_time"])
        arrival_time, arr_next_day = parse_time(row["arrival_time"])
        line = row["line"]

        graph[start_stop].append((departure_time, arrival_time, line, end_stop, dep_next_day, arr_next_day))

    return graph


# Przykładowe użycie
graph = load_graph("connection_graph.csv")

