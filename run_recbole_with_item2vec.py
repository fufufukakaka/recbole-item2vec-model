from logging import getLogger

import click
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from models.Item2Vec import Item2Vec


@click.command()
@click.option(
    "-d",
    "--dataset_name",
    required=True,
    type=str,
    help="Dataset Name",
)
@click.option(
    "-c",
    "--config_files",
    required=True,
    type=str,
    help="config file path",
)
def main(dataset_name, config_files):
    config_file_list = config_files.strip().split(" ") if config_files else None
    parameter_dict = {"neg_sampling": None, "stopping_step": 1}
    config = Config(
        model=Item2Vec,
        dataset=dataset_name,
        config_file_list=config_file_list,
        config_dict=parameter_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    model = Item2Vec(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info("best valid result: {}".format(best_valid_result))
    logger.info("test result: {}".format(test_result))


if __name__ == "__main__":
    main()
