import indicators as id
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from ManualStrategy import stats
from StrategyLearner import StrategyLearner
from experiment1 import experiment1
from experiment2 import experiment2
import random as rand
rand.seed(902107089)


def author():
        return "zelahmadi3"

if __name__ == "__main__":
    experiment1()
    experiment2()
    stats()