import logging

from tensorboard import program

from config import current_config

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logging.getLogger("matplotlib").setLevel(logging.ERROR)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(current_config.tensorboard_path)])
    url = tb.launch()
    logging.info(f"Tensorflow listening on {url}")
    tb.main()
