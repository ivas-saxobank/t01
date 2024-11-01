from typing import Sequence, Union, Literal
from collections import OrderedDict

import gymnasium as gym

import datetime
date = datetime.datetime.now()

year_start = date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
year_end = date.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)

iso_day = date.isoweekday()
time = (date.timestamp() - day_start.timestamp()) / float(24 * 60 * 60)

wtday =  (iso_day - 1 + time) / 7
print(wtday, iso_day, time)