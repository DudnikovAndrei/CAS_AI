import os, os.path, requests, datetime
import numpy as np
import pandas as pd
import bs4 as bs
import yfinance

from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

import torch, torchmetrics
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')