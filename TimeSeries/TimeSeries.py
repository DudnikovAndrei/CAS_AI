from pytorch_forecasting import TimeSeriesDataSet, Baseline

class TimeSeries():
    def __init__(self):
        self.max_prediction_length = 6
        self.max_encoder_length = 24

    def get_training(self, train_dataset, features_columns):
        training_cutoff = train_dataset["time_idx"].max() - self.max_prediction_length
        training = TimeSeriesDataSet(
            train_dataset[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="SPY",
            group_ids=["group_id"],
            min_encoder_length=self.max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,

            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=features_columns,

            # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        return training

    def get_validation(self, training, train_dataset):
        return TimeSeriesDataSet.from_dataset(training, train_dataset, predict=True, stop_randomization=True)

    def get_train_Dataloader(self, training, batch_size):
        return training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

    def get_val_dataloader(self, validation, batch_size):
        return validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    def get_baseline_predictions(self, val_dataLoader):
        return Baseline().predict(val_dataLoader)