# Generic imports
import math
from sklearn import metrics

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.nn        import *

###############################################
### NN prediction on glass dataset
class nn_glass_mig_density(base_app):
    def __init__(self):

        # Main parameters
        self.name = "nn_glass_mig_density"

        # Initialize mother class
        super().__init__(self.name)

    def run(self):

        # Load dataset
        dts = dataset(pms(name="glass_mig_density",
                          path="C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density",
                          data=pms(type="array",
                                   filename="data_mig_density.csv",
                                   skiprows=1),
                          labels=pms(type="array",
                                     filename="labels_mig_density.csv",
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
                               arch=[2000,2000,2000,1],
                               acts=["relu","relu","relu","linear"]),
                     loss="mse",
                     lr=1.0e-4))

        # Convert dataset to torch tensors
        dts.to_torch()

        # Train
        loss = net.train(dts, 5000, 0.5)

        # Plot
        filename = "C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/results/nn_glass_mig/loss.png"
        plot_training(filename, loss[:,0], loss[:,1], loss[:,2], log_y=True)

        # Apply to test set
        xt, yt = dts.test_data()
        yp = net.apply(xt)
        
        # SiO2 Na2O Al2O3 B2O3 CaO MgO K2O Li2O PbO BaO ZnO
        entree_manuelle = [73, 15, 1, 0, 7, 4, 0, 0, 0, 0, 0]   #verre microscope
        mean = np.mean(entree_manuelle)
        std = np.std(entree_manuelle)
        entree_manuelle = (entree_manuelle - mean)/std
        entree_manuelle = torch.Tensor(entree_manuelle)
        
        sortie_manuelle = net.apply(entree_manuelle)
        
        print("Densité verre microscope :", sortie_manuelle)
        # Plot predicted density again label
        filename = "C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/results/nn_glass_mig/comparison.png"
        y = np.hstack((yt, yp))
        e = np.abs(yt-yp)
        r_sqr = metrics.r2_score(yt, yp)
        plot_scatter(filename, y,
                     title="True density against prediction (test set), R² = " + str(round(r_sqr, 3)), c=e)
        print("# Average error on test set: "+str(np.mean(e.numpy())))

        # Save network parameters
        
        net.dump("C:/Users/adelie.saule/MIG/Courbe_évolution_densité_SiO2_B2O3/net_density.txt")