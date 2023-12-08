# Generic imports
import math

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.pca       import *

###############################################
### PCA of glass dataset
class pca_glass(base_app):
    def __init__(self):

        # Main parameters
        self.name = "pca_glass"

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

        # Plot repartition
        filename = self.results_path+"/repartition.png"
        plot_continuous_bins(filename, dts.labels().x(), 30)

        # Normalize
        dts.normalize_data(std=False)

        # Load and train pca
        p = pca(npc=2, dts=dts)
        p.train()
        p.interpret(limit=0.5)

        # Apply pca to dataset
        x = p.apply(dts.data())

        # Plot with density label
        filename = self.results_path+"/pca_density.png"
        plot_scatter(filename, x,
                     title="PCA(pc=2) of glass dataset, density",
                     c=dts.labels().x())

        # Plot with certain oxydes richness
        xd = dts.data().x().copy()
        xd = dts.denormalize_data(xd)

        filename = self.results_path+"/pca_sio2.png"
        plot_scatter(filename, x,
                     title="PCA(pc=2) of glass dataset, SiO2",
                     c=xd[:,0])

        filename = self.results_path+"/pca_cao.png"
        plot_scatter(filename, x,
                     title="PCA(pc=2) of glass dataset, CaO",
                     c=xd[:,3])

        filename = self.results_path+"/pca_na2o.png"
        plot_scatter(filename, x,
                     title="PCA(pc=2) of glass dataset, Na2O",
                     c=xd[:,5])

        # Reconstruct
        xx, err = p.reconstruct(x)

        # Redo with more components
        p = pca(npc=4, dts=dts)
        p.train()
        x       = p.apply(dts.data())
        xx, err = p.reconstruct(x)
