import numpy as np
import awkward as ak
import uproot


file = uproot.open("../data/cern/fatjet-processes/H_QCD.root")
print(file.keys())
file = uproot.open("../data/cern/fatjet-processes/H_QCD.root")

tree = file["outnano/Jets"]  # uproot ignores ;1 by default

tree.show()
