# Generic imports
import math

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.nn        import *

###############################################
### NN prediction on diabetes dataset
class nn_diabetes(base_app):
    def __init__(self):

        # Main parameters
        self.name = "nn_diabetes"

        # Initialize mother class
        super().__init__(self.name)

    def run(self):

        # Load dataset
        dts = dataset(pms(name="diabetes",
                          path="hummingbird/dts/diabetes",
                          data=pms(type="array",
                                   filename="data.csv",
                                   skiprows=1),
                          labels=pms(type="array",
                                     filename="labels.csv",
                                     skiprows=1)
                          )
                      )
        dts.load()

        # Normalize
        dts.normalize_data()

        # Split
        dts.split(0.2, 0.0, random=True)

        # Load nn
        net = nn(pms(model=pms(inp_dim=dts.n_features(),
                               arch=[16,16,2],
                               acts=["tanh","tanh","tanh"]),
                     loss="crossentropy",
                     lr=1.0e-4))

        # Convert dataset to torch tensors
        dts.to_torch()

        # Train
        loss = net.train(dts, 2000, 0.5)

        # Plot
        filename = self.results_path+"/loss.png"
        plot_training(filename, loss[:,0], loss[:,1], loss[:,2], log_y=True)
