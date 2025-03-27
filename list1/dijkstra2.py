import heapq
from datetime import datetime, timedelta

from list1.data_parser import graph

TRANSFER_TIME = timedelta(hours=0, minutes=1, seconds=0)

def time_to_minutes(time_obj, next_day=False):
    """ Konwertuje obiekt datetime.time na liczbę minut od północy. Obsługuje następny dzień. """
    minutes = time_obj.hour * 60 + time_obj.minute
    if next_day:
        minutes += 24 * 60  
    return minutes

def dijkstra_shortest_time(graph, start, end, start_time):
    """
    Znajduje trasę o najkrótszym czasie przejazdu między start a end za pomocą algorytmu Dijkstry.
    :param graph: Graf przystanków w postaci słownika
    :param start: Nazwa przystanku początkowego
    :param end: Nazwa przystanku docelowego
    :param start_time: Godzina przybycia na przystanek początkowy (obiekt datetime.time)
    :return: Najkrótsza trasa jako lista (czas odjazdu, czas przyjazdu, linia, przystanek początkowy, przystanek końcowy)
    """
    pq = []  # Kolejka priorytetowa (czas przyjazdu, przystanek, linia, trasa)
    heapq.heappush(pq, (time_to_minutes(start_time), start, None, [], []))  # None = brak wcześniejszej linii

    visited = {}

    while pq:
        current_time, current_stop, current_line, path, last_line = heapq.heappop(pq)

        if current_stop in visited and visited[current_stop] <= current_time:
            continue
        visited[current_stop] = current_time

        if current_stop == end:
            return path

        for dep_time, arr_time, line, next_stop, dep_next_day, arr_next_day in graph[current_stop]:
            dep_minutes = time_to_minutes(dep_time, dep_next_day)
            arr_minutes = time_to_minutes(arr_time, arr_next_day)


            # Dodajemy minutę na przesiadkę, jeśli zmieniamy linię
            transfer_time = int(TRANSFER_TIME.total_seconds() / 60) if last_line and last_line != line else 0

            if dep_minutes >= current_time+transfer_time:  # Możemy odjechać tylko później niż obecny czas
                new_path = path + [(dep_time, arr_time, line, current_stop, next_stop)]
                heapq.heappush(pq, (arr_minutes, next_stop, line, new_path, line))

    return None  # Brak trasy

# Przykładowe wywołanie
start = "Zajezdnia Obornicka"
end = "PORT LOTNICZY"
start_time = datetime.strptime("20:50:00", "%H:%M:%S").time()

shortest_path = dijkstra_shortest_time(graph, start, end, start_time)

if shortest_path:
    print("Najkrótsza trasa według czasu przejazdu:")
    for dep, arr, line, from_stop, to_stop in shortest_path:
        print(f"Linia {line}: {from_stop} ({dep}) → {to_stop} ({arr})")
else:
    print("Brak możliwej trasy.")
