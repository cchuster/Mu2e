#%% Necessary imports
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
# Path to the desired ROOT file
files = ["trkana.half.root:TrkAnaExt/trkana"]
#%% Calculating the path length of each reconstructed track on the left and right halves of a differentially calibrated detector
# Selecting the data fields we want to use, namely the distance of closest approach, the acceptability of the linear fit, the cosine of the angle between the wire and the local track direction, and the plane
data_fields = ["rdoca", "state", "wdot", "plane", "edep"]
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
#%% Plot edep vs pathLength (expect a positive correlation)
plt.figure(figsize=(10, 6))
plt.scatter(a["edep"][hitCut], pathLength, alpha=0.5, label='edep vs pathLength')
plt.xlabel('edep')
plt.ylabel('pathLength')
plt.title('edep vs pathLength')
plt.legend()
plt.grid(True)
plt.show()
#%% Plot pathLength as a histogram (expect a gaussian)
plt.figure(figsize=(10, 6))
plt.hist(pathLength, range=(0,4), bins=100, alpha=0.5, label='pathLength')
plt.xlabel('Path Length')
plt.ylabel('Frequency')
plt.title('Histogram of Path Length')
plt.legend()
plt.grid(True)
plt.show()
#%% Plot dE/dx for left and right halves
planeCut = a["plane"][hitCut] < 18
pathCut = pathLength >= 1.3
de = a["edep"][hitCut]
dedxLeft = de[planeCut & pathCut]/pathLength[planeCut & pathCut]
dedxRight = de[~planeCut & pathCut]/pathLength[~planeCut & pathCut]
print(dedxLeft)
print(dedxRight)

plt.figure(figsize=(10, 6))
plt.hist(dedxLeft, range=(0.000,0.006), bins=100, alpha=0.5, label='dedxLeft', density=True)
plt.hist(dedxRight, range=(0.000,0.006), bins=100, alpha=0.5, label='dedxRight', density=True)
plt.xlabel('dE/dx')
plt.ylabel('Density')
plt.title('Histogram of dE/dx for Left and Right Halves')
plt.legend()
plt.grid(True)
plt.xlim(0.00, 0.006)
plt.show()

# Find the bin that contains the most data points (i.e. find the mode of continuous data, since usually no one exact value occurs more than once)
histLeft, binsLeft = np.histogram(dedxLeft, bins=100, range=(0.000, 0.006))
histRight, binsRight = np.histogram(dedxRight, bins=100, range=(0.000, 0.006))
modeLeft = binsLeft[np.argmax(histLeft)]
modeRight = binsRight[np.argmax(histRight)]
print(f"Mode of dedxLeft: {modeLeft}")
print(f"Mode of dedxRight: {modeRight}")
# %%
