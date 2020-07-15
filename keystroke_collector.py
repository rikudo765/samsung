from pynput.keyboard import Key, Listener
import numpy as np
import time
import pandas as pd

global start_pr
global start_int
global data
global h


SESSION = 2

data = []


def on_press(key):
    global start_pr
    global h
    global check
    print("{} pressed".format(key))
    start_pr = time.time()
    if check:
        check = False
        return
    UD = time.time() - start_int
    raw.append(h + UD)
    raw.append(UD)
    if key == Key.enter:
        # Stop listener
        return


def on_release(key):
    global start_pr
    global start_int
    global h
    h = time.time() - start_pr
    raw.append(h)
    if key == Key.enter:
        # Stop listener
        return False
    # duration of button press
    start_int = time.time()


for i in range(15):
    time.sleep(2)
    raw = [SESSION, i + 1]
    check = True
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        print("Password: ")
        listener.join()
    data.append(raw)

df = pd.DataFrame(np.array(data), columns=["Session", "attempt", "H.", "DD.", "UD.", "Ht", "DDt", "UDt", "Hi", "DDi",
                                           "UDi", "He", "DDe", "UDe", "H5", "DD5", "UD5", "Hshift", "DDshift",
                                           "UDshift", "Hr", "DDr", "UDr", "Ho", "DDo", "UDo", "Ha", "DDa", "UDa",
                                           "Hn", "DDn", "UDn", "Hl", "DDl", "UDl", "Henter"])
df.to_csv("dataset.csv", index=False, encoding='utf-8', mode="a")
