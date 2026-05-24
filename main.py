import warnings
warnings.filterwarnings('ignore')

import core.imports
from core.engine import Engine


if __name__ == "__main__":
    engine = Engine("config.yaml")
    engine.run()