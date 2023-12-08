# Generic imports
import math

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.nn        import *

###############################################
### NN prediction on glass dataset
class nn_glass(base_app):
    def __init__(self):

        # Main parameters
        self.name = "nn_glass"

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
        dts.split(0.1, 0.1, random=True)

        # Load nn
        net = nn(pms(model=pms(inp_dim=dts.n_features(),
                               arch=[32,32,1],
                               acts=["relu","relu","linear"]),
                     loss="mse",
                     lr=1.0e-3))

        # Convert dataset to torch tensors
        dts.to_torch()

        # Train
        loss = net.train(dts, 2500, 0.5)

        # Plot
        filename = self.results_path+"/loss.png"
        plot_training(filename, loss[:,0], loss[:,1], loss[:,2], log_y=True)

        # Apply to test set
        xt, yt = dts.test_data()
        yp = net.apply(xt)

        # Plot predicted density again label
        filename = self.results_path+"/comparison.png"
        y = np.hstack((yt, yp))
        e = np.abs(yt-yp)
        plot_scatter(filename, y,
                     title="True density against prediction (test set)", c=e)
        print("# Average error on test set: "+str(np.mean(e.numpy())))
