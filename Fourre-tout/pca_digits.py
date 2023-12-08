# Custom imports
from hummingbird.app.base_app import *

###############################################
### PCA of digits dataset
class pca_digits(base_app):
    def __init__(self):

        # Main parameters
        self.name = "pca_digits"

        # Initialize mother class
        super().__init__(self.name)

    def run(self):

        # Load dataset
        dts = dataset(pms(name="digits",
                          path="hummingbird/dts/digits",
                          data=pms(type="image",
                                   source="csv",
                                   nx=8,
                                   ny=8,
                                   nz=1,
                                   filename="data.csv"),
                          labels=pms(type="array",
                                     filename="labels.csv",
                                     dtype=int)
                          )
                      )
        dts.load()

        # Plot dataset examples
        filename = self.results_path+"/data.png"
        plot_mosaic(filename, dts.data().x(), 10, 5)

        # Plot repartition
        filename = self.results_path+"/repartition.png"
        bins = np.arange(11) - 0.5
        ticks = np.arange(10)
        plot_discrete_bins(filename, dts.labels().x(), bins, ticks)

        # Normalize
        dts.normalize_data(std=False)

        # Load and train pca
        p = pca(npc=2, dts=dts)
        p.train()

        # Apply pca to dataset
        x = p.apply(dts.data())

        # Plot
        filename = self.results_path+"/pca.png"
        plot_scatter(filename, x, title="PCA(pc=2) of digits dataset",
                     c=dts.labels().x(),
                     cmap=plt.cm.get_cmap('Spectral', 10))

        # Reconstruct
        xx, err  = p.reconstruct(x)
        filename = self.results_path+"/reconstruct_2.png"
        plot_mosaic(filename, xx, 10, 5)

        # Redo with more components
        p = pca(npc=20, dts=dts)
        p.train()
        x       = p.apply(dts.data())
        xx, err = p.reconstruct(x)
        filename = self.results_path+"/reconstruct_10.png"
        plot_mosaic(filename, xx, 10, 5)
