# Generic imports
import math

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.ae        import *

###############################################
### AE of glass dataset
class ae_glass(base_app):
    def __init__(self):

        # Main parameters
        self.name = "ae_glass"

        # Initialize mother class
        super().__init__(self.name)

    def run(self):

        # Load dataset
        dts = dataset(pms(name="glass",
                          path="hummingbird/dts/glass",
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

        # Load and train pca
        inp_dim = dts.n_features()
        net = ae(pms(model=pms(inp_dim=inp_dim,
                               lat_layer=2,
                               arch=[8,4,2,4,8,inp_dim],
                               acts=["relu","relu","relu","relu","relu","relu"]),
                     loss="mse",
                     lr=1.0e-3))

        # Convert dataset to torch tensors
        dts.to_torch()

        # Train
        loss = net.train(dts, 5000, 0.5)

        # Apply ae to dataset
        xl, xp = net.apply(dts.data())

        # Plot
        filename = self.results_path+"/ae.png"
        plot_scatter(filename, xl, title="AE(dim=2) of glass dataset",
                     c=dts.labels().x())
