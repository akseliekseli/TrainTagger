# flake8: noqa
try:
    from tagger.model.DeepSetModel import DeepSetModel
except ImportError:
    DeepSetModel = None

try:
    from tagger.model.DeepSetModelHGQ import DeepSetModelHGQ
except ImportError:
    DeepSetModelHGQ = None

try:
    from tagger.model.InteractionNetModel import InteractionNetModel
except ImportError:
    InteractionNetModel = None

try:
    from tagger.model.TransformerModel import TransformerModel
except ImportError:
    TransformerModel = None

try:
    from tagger.model.DeepSetModelSGD import DeepSetModelSGD
except ImportError:
    DeepSetModelSGD = None

try:
    from tagger.model.TransformerFatJet import TransformerFatJet
except ImportError:
    TransformerFatJet = None

try:
    from tagger.model.DeepSetModelMulti import DeepSetModelMulti
except ImportError:
    DeepSetModelMulti = None

try:
    from tagger.model.CascadeModel import CascadeModel
except ImportError:
    CascadeModel = None

try:
    from tagger.model.AdversarialDeepSetModel import AdversarialDeepSetModel
except ImportError:
    AdversarialDeepSetModel = None

try:
    from tagger.model.DeepSetModelMD import DeepSetModelMD
except ImportError:
    DeepSetModelMD = None
