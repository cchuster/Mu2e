#%%
import uproot
import awkward as ak
import numpy as np
# Path to the desired ROOT file
files = ["trkana.half.root:TrkAnaExt/trkana"]
#%% Calculating the path length of each reconstructed track on the left and right halves of the differentially calibrated detector
# Selecting the data fields we want to use, namely the distance of closest approach, the acceptability of the linear fit, the cosine of the angle between the wire and the local track direction, and the plane
data_fields = ["rdoca", "state", "wdot", "plane"]
# Create a dictionary that stores each of these fields as keys with an associated list of values
a = {field: [] for field in data_fields}

for batch in uproot.iterate(files, filter_name="/kl|kltsh/i"):
    # Create an awkward array that is True when the fit is good (i.e. status==1)
    cutStatus = ak.sum(batch["kl.status"], axis=1) == 1
    # For each data field, flatten and append the values of the awkward array to their corresponding lists and convert them to NumPy arrays
    for field in data_fields:
        a[field].append(ak.flatten(ak.flatten(batch["kltsh"][field][cutStatus])).to_numpy())

# Create an empty array that will store the final path lengths of each track
pathLength = []

for field in a:
    # The arrays above were parsed in batches, so we concatenate them all together into a single NumPy array
    a[field] = np.concatenate(a[field])
    print(f"Dimensions of a[{field}]: {a[field].shape}")
# Make another filter such that we only consider the tracks with recorded radial position (i.e. state==1) and with radius values that are legitimate (i.e. are bounded by the 2.5mm radius of the straw)
hitCut = (np.abs(a["state"]) == 1) & (np.abs(a["rdoca"]) < 2.5)
# Calculate the path length inside the straw using the formula below derived via the Pythagorean Theorem and basic angle relations    
pathLength = np.abs((np.sqrt((6.25 - (a["rdoca"][hitCut])**2) / 4)) / (np.sin(np.arccos((a["wdot"][hitCut])))))
print(f"Dimensions of pathLength: {pathLength.shape}")
print(pathLength)

# Calculate the mean path length for the left and right halves of the detector using a boolean mask
meanLeft = np.mean(pathLength[a["plane"][hitCut] < 18])
meanRight = np.mean(pathLength[a["plane"][hitCut] >= 18])
print(f"Mean of pathLengthLeft: {meanLeft}")
print(f"Mean of pathLengthRight: {meanRight}")





# %%
# #%% Finding the edep means of each half of a differentially calibrated detector
# # Get the edep leaf from the trkana tree
# data_fields = ["edep"]
# mc_fields = ["edep"]

# # Create a dictionary with key as field and value as list of entries in edep
# aLeft = {field: [] for field in data_fields}
# aRight = {field: [] for field in data_fields}
# amc = {field: [] for field in mc_fields}

# # For each batch of data that uproot automatically generates based on available RAM...
# for batch in uproot.iterate(files,filter_name="/kl|kltsh|kltshmc/i"):
#     # Selecting the hits from each half that are good linear fits i.e. abs(status) = 1
#     cutLeft = (ak.sum(batch["kl.status"], axis=1) == 1) & (batch["kltsh"]["plane"] < 18)
#     cutRight = (ak.sum(batch["kl.status"], axis=1) == 1) & (batch["kltsh"]["plane"] >= 18)
#     # First flatten the events, then flatten the tracks, then flatten and append the hits from these tracks to the keys
#     # Can use type.show() to show the dimensions of the object being used
#     for field in data_fields:
#         aLeft[field].append(ak.flatten(ak.flatten(batch["kltsh"][field][cutLeft])).to_numpy())
#         aRight[field].append(ak.flatten(ak.flatten(batch["kltsh"][field][cutRight])).to_numpy())
#     # Do the same for monte carlo truth values but only include elements with indices less than the number of elements in the kltsh branch (kltsh branch has less data points because of threshold)
#     for field in mc_fields:
#         amc[field].append(ak.flatten(ak.flatten(batch["kltshmc"][field][ak.local_index(batch["kltshmc"][field]) < ak.num(batch["kltsh"][data_fields[0]],axis=2)])).to_numpy())

# # Concatenate the lists of NumPy arrays generated from each batch
# # The length of the array is just the number of tracks we simulated times the number of hits in each track
# for field in aLeft:
#     aLeft[field] = np.concatenate(aLeft[field])
# for field in aRight:
#     aRight[field] = np.concatenate(aRight[field])
# for field in amc:
#     amc[field] = np.concatenate(amc[field])

# # Find the mean of a[field] and amc[field]
# mean_aLeft = np.mean(aLeft[field])
# mean_aRight = np.mean(aRight[field])
# print("<18 mean: " + str(mean_aLeft), ">=18 mean: " + str(mean_aRight), "diff: " + str(abs(mean_aLeft - mean_aRight)))