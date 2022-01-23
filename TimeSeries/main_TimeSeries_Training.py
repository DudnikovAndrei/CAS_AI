import torch
from SP500_TimeSeries import SP500DataSet
from TimeSeries import TimeSeries
from TimeSeries_Trainer import TimeSeriesTrainer
from pytorch_forecasting import TemporalFusionTransformer
from HyperParams_Tuning import HyperParamsTuning

#
# Configs
#
nTest = 1000
batch_size = 128  # set this between 32 to 128

#
# Data Set
#
sp500_dataset = SP500DataSet()
timeSeries = TimeSeries()

train_dataset = sp500_dataset.get_train_dataset(nTest)
features_columns = list(train_dataset.columns.values[:-4])

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
training = timeSeries.get_training(train_dataset, features_columns)
validation = timeSeries.get_validation(training, train_dataset)

# create dataloaders for model
train_dataloader = timeSeries.get_train_Dataloader(training, batch_size)
val_dataloader = timeSeries.get_val_dataloader(validation, batch_size)

#
# Base Line Model
#

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = timeSeries.get_baseline_predictions(val_dataloader)
print((actuals - baseline_predictions).abs().mean().item())


#
# Train the Temporal Fusion Transformer
#
# Find optimal learning rate
timeSeriesTrainer = TimeSeriesTrainer()
trainer_for_lr = timeSeriesTrainer.get_trainer_for_lr()
tft_for_lr = timeSeriesTrainer.get_tft_for_lr(training)
lr = timeSeriesTrainer.find_lr(trainer_for_lr, tft_for_lr, train_dataloader, val_dataloader)

# Train Model
trainer = timeSeriesTrainer.get_trainer()
tft = timeSeriesTrainer.get_tft(training)

# fit network
trainer.fit(tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

#
# Hyperparameter tuning
#
HyperParamsTuning().create_study(train_dataloader, val_dataloader)

#
# Evaluate performance
#

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
