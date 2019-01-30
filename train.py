import torch
from torch import nn, optim
import numpy as np
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy
from time import time
from datetime import timedelta
from pathlib import Path
from nicetable.nicetable import NiceTable


def train(model, model_vec, data_train, data_test, config, model_name):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    trainer = create_supervised_trainer(
        model, optimizer, criterion
    )

    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "accuracy": Accuracy()
        }  # ,
        # non_bloking=True
    )

    @trainer.on(Events.STARTED)
    def start_prepare(trainer):
        trainer._loss = []
        trainer._tmp_loss = []
        trainer._last_loss = None
        trainer._accuracy_train = []
        trainer._accuracy_test = []
        trainer._init_time = time()
        trainer._csv_path = config["log_path"]

        if not Path(trainer._csv_path):
            with open(trainer._csv_path, "x") as f:
                f.write("epoch,loss,accuracy_train,accuracy_test,model\n")
                f.close()

    @trainer.on(Events.ITERATION_COMPLETED)
    def mean_loss(trainer):
        trainer._tmp_loss.append(trainer.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def registry_progress(trainer):

        trainer._last_loss = np.mean(trainer._tmp_loss)
        trainer._tmp_loss = []

        loss = trainer._last_loss
        trainer._loss.append(
            loss
        )

        evaluator.run(data_train)
        accuracy_train = evaluator.state.metrics["accuracy"]
        trainer._accuracy_train.append(
            accuracy_train
        )

        evaluator.run(data_test)
        accuracy_test = evaluator.state.metrics["accuracy"]
        trainer._accuracy_test.append(
            accuracy_test
        )

        with open(trainer._csv_path, "a") as f:
            f.write(
                f"{trainer.state.epoch},{loss},{accuracy_train},{accuracy_test},{model_name}\n")
            f.close()

    @trainer.on(Events.EPOCH_COMPLETED)
    def progress_log(trainer):
        if trainer.state.epoch % 5 == 0:
            evaluator.run(data_train)
            metrics = evaluator.state.metrics
            print(f"epoch: {trainer.state.epoch:<2} | " +
                  f"accuracy: {metrics['accuracy']:.2f} | " +
                  f"loss: {trainer._loss[-1]:.2f}")

    @trainer.on(Events.COMPLETED)
    def summary(trainer):
        print("-"*80)
        print(f"model: {model_name}")
        print(f"epochs: {trainer.state.epoch}")
        print(f"total time: {timedelta(seconds=(time() - trainer._init_time))}")

        out = NiceTable(["--", "loss", "train", "test"])
        out.append(["init",
                    f"{trainer._loss[0]:.3f}",
                    f"{trainer._accuracy_train[0]:.3f}",
                    f"{trainer._accuracy_test[0]:.3f}"])
        out.append(["end",
                    f"{trainer._loss[-1]:.3f}",
                    f"{trainer._accuracy_train[-1]:.3f}",
                    f"{trainer._accuracy_test[-1]:.3f}"])
        print(out)

        torch.save(model, f"saved_models/{model_name}")

    trainer.run(data_train, max_epochs=config["epochs"])
    return trainer
