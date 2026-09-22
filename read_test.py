import uproot
f = uproot.open("/eos/home-a/asuutari/training_data/data_chunk_0.root")
print(f["data"].classname)
obj = f["data"]
arrays = obj.arrays()  # try reading directly
print(arrays.fields)
