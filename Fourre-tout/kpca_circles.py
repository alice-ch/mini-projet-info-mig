# Generic imports
import math

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.circles import *
from hummingbird.src.agent.kpca      import *
from hummingbird.src.agent.pca       import *

###############################################
### KPCA of circles dataset
class kpca_circles(base_app):
    def __init__(self):

        # Main parameters
        self.name = "kpca_circles"

        # Initialize mother class
        super().__init__(self.name)

    def run(self):

        # Load dataset
        dts = circles(pms(n_radii=2,
                          radii=[1.0, 0.3],
                          n_samples_per_radius=200,
                          noise=0.02))

        # Plot dataset
        filename = self.results_path+"/circles.png"
        plot_scatter(filename, dts.data().x(), title="Circles dataset",
                     c=dts.labels().x())

        # Normalize
        # We don't normalize for kpca, as the kernel matrix will
        # be centered and normalized

        # Load and train kpca
        p = kpca(npc=2, dts=dts, gamma=2.0)
        x = p.train()

        # Plot
        filename = self.results_path+"/kpca.png"
        plot_scatter(filename, x, title="KPCA(pc=2) of circles dataset",
                     c=dts.labels().x())

        y = p.apply(dts.data())
        filename = self.results_path+"/ykpca.png"
        plot_scatter(filename, y, title="KPCA(pc=2) of circles dataset",
                     c=dts.labels().x())

        # Create a new dataset with different radii
        dts2 = circles(pms(n_radii=2,
                           radii=[0.2, 0.7],
                           n_samples_per_radius=200,
                           noise=0.04))

        # Project on kpca basis from first dataset
        y = p.apply(dts2.data())

        # Stack both projections
        xx = np.vstack((x, y))
        xl = np.vstack((dts.labels().x(), dts2.labels().x()+2))

        # Plot
        filename = self.results_path+"/kpca2.png"
        plot_scatter(filename, xx, title="KPCA(pc=2) of circles dataset",
                     c=xl)

        # Normalize and apply pca for comparison
        dts.normalize()
        p = pca(npc=2, dts=dts)
        p.train()
        x = p.apply(dts.data())

        # Plot
        filename = self.results_path+"/pca.png"
        plot_scatter(filename, x, title="PCA(pc=2) of circles dataset",
                     c=dts.labels().x())
