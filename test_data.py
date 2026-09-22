import uproot

filename = "/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_191125_151X/GluGluHHTo4B_PU200/FP/fp_jettuples_v151Xv1/jetTuple_extended_5_8776563_0.root"
tree = uproot.open(filename)["outnano/Jets"]

for field in tree.keys():
    print(field)
