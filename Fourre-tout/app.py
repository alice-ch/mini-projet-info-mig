# Custom imports
from hummingbird.src.core.factory import *
from hummingbird.app.pca_glass    import *
from hummingbird.app.pca_digits   import *
from hummingbird.app.kpca_circles import *
from hummingbird.app.nn_diabetes  import *
from hummingbird.app.nn_glass     import *
from hummingbird.app.ae_glass     import *
from hummingbird.app.nn_glass_mig_density import *
from hummingbird.app.nn_glass_mig_young import *
from hummingbird.app.nn_glass_mig_vickers import *
from hummingbird.app.nn_glass_mig_density_v2 import *
from hummingbird.app.nn_glass_mig_young_v2 import *
from hummingbird.app.nn_glass_mig_vickers_v2 import *
from hummingbird.app.nn_glass_mig_hardness_v2 import *

# Declare factory
app_factory = factory()

# Register apps
app_factory.register("pca_glass",    pca_glass)
app_factory.register("pca_digits",   pca_digits)
app_factory.register("kpca_circles", kpca_circles)
app_factory.register("nn_diabetes",  nn_diabetes)
app_factory.register("nn_glass",     nn_glass)
app_factory.register("ae_glass",     ae_glass)
app_factory.register("nn_glass_mig_density", nn_glass_mig_density)
app_factory.register("nn_glass_mig_young", nn_glass_mig_young)
app_factory.register("nn_glass_mig_vickers", nn_glass_mig_vickers)
app_factory.register("nn_glass_mig_density_v2", nn_glass_mig_density_v2)
app_factory.register("nn_glass_mig_young_v2", nn_glass_mig_young_v2)
app_factory.register("nn_glass_mig_vickers_v2", nn_glass_mig_vickers_v2)
app_factory.register("nn_glass_mig_hardness_v2", nn_glass_mig_hardness_v2)