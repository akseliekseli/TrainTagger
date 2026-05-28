import uproot
import awkward as ak

# Check merged file
with uproot.open("../data/cern/bbccqq_qcd.root") as f:
    tree = f["outnano/Jets"]
    labels = tree["fj_label"].array(entry_stop=5)
    print("Merged:")
    print(labels)
    print(labels.type)

# Check original file
with uproot.open("../data/cern/hbb/GluGluHHTo4B_PU200.root") as f:
    tree = f["outnano/Jets"]
    labels = tree["fj_label"].array(entry_stop=5)
    print("Original:")
    print(labels)
    print(labels.type)
