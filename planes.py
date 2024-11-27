#%% Necessary imports
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
# Path to the desired ROOT file
files = ["/home/online1/rbonventre/energycalibration/trkana.planes.0.root"]
#%% Fields of interest to calibrate simulated edep
data_fields = ["rdoca", "state", "wdot", "plane", "edep"]
a = {field: [] for field in data_fields}
amc = {"edep": []}
#%% Gather the simulation and mc data
for batch in uproot.iterate(files, filter_name="/kl|kltsh|kltshmc/i"):
    # Only consider data from good fits
    cutStatus = ak.sum(batch["kl.status"], axis=1) == 1
    for field in data_fields:
        a[field].append(ak.flatten(ak.flatten(batch["kltsh"][field][cutStatus])).to_numpy())
    amc["edep"].append(ak.flatten(ak.flatten(batch["kltshmc"]["edep"][ak.local_index(batch["kltshmc"]["edep"]) < ak.num(batch["kltsh"]["edep"],axis=2)][cutStatus])).to_numpy())
    break

for field in a:
    # The arrays above were parsed in batches, so we concatenate them all together into a single NumPy array
    a[field] = np.concatenate(a[field])
    print(f"Dimensions of a[{field}]: {a[field].shape}")
amc["edep"] = np.concatenate(amc["edep"])
print(f"Dimensions of amc['edep']: {amc['edep'].shape}")
#%% Finding path length
# Create an empty array that will store the final path lengths of each track
pathLength = []
# Make another filter such that we only consider the tracks with recorded radial position (i.e. state==1) and with radius values that are legitimate (i.e. are bounded by the 2.5mm radius of the straw)
hitCut = (np.abs(a["state"]) == 1) & (np.abs(a["rdoca"]) < 2.5)
# Calculate the path length inside the straw using the formula below derived via the Pythagorean Theorem and basic angle relations    
pathLength = np.abs((np.sqrt((6.25 - (a["rdoca"][hitCut])**2) / 4)) / (np.sin(np.arccos((a["wdot"][hitCut])))))
print(f"Dimensions of pathLength: {pathLength.shape}")
print(pathLength)

# Find the mean path lengths for each track in all 36 individual planes
meanPathLengths = []
for plane in range(36):
    planeCut = a["plane"][hitCut] == plane
    meanPathLength = np.mean(pathLength[planeCut])
    meanPathLengths.append(meanPathLength)
    print(f"Mean of pathLength for Plane {plane}: {meanPathLength}")
#%% Finding dE/dx for each plane
pathCut = pathLength >= 1.3
de = a["edep"][hitCut]
planes = a["plane"][hitCut]

dedxPerPlane = []
for plane in range(36):
    planeCut = planes == plane
    dedx = de[planeCut & pathCut]/pathLength[planeCut & pathCut]
    dedxPerPlane.append(dedx)

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(dedxPerPlane)))
for plane, dedx in enumerate(dedxPerPlane):
    plt.hist(dedx, range=(0.000,0.006), bins=100, alpha=0.5, label=f'dedxPlane{plane}', density=True, color=colors[plane])
plt.xlabel('dE/dx')
plt.ylabel('Density')
plt.title('Histogram of dE/dx for Each Plane')
plt.legend()
plt.grid(True)
plt.xlim(0.00, 0.006)
plt.show()
# Calculate the average dE/dx value for each plane
average_dedx_per_plane = []
for plane in range(36):
    planeCut = a["plane"][hitCut] == plane
    pathCut = pathLength >= 1.3
    de = a["edep"][hitCut]
    path = pathLength
    dedx = de[planeCut & pathCut]/path[planeCut & pathCut]
    average_dedx = np.mean(dedx)
    average_dedx_per_plane.append(average_dedx)
    print(f"Average dE/dx for Plane {plane}: {average_dedx}")
# %% Scaling of edep values
# Calculate the target mean (the mean we are calibrating towards), namely the mean of the means of each plane found above
target_mean = np.mean(average_dedx_per_plane)
print(f"Target Mean (Mean of Means of Each Plane): {target_mean}")
# Divide each entry in average_dedx_per_plane by the target mean
normalized_average_dedx_per_plane = [average_dedx / target_mean for average_dedx in average_dedx_per_plane]
# Scale all edep values of each plane by the normalized_average_dedx_per_plane
scaled_edep_per_plane = []
for plane in range(36):
    planeCut = a["plane"][hitCut] == plane
    de = a["edep"][hitCut]
    scaled_de = de[planeCut] / normalized_average_dedx_per_plane[plane]
    scaled_edep_per_plane.append(scaled_de)
    print(f"Scaled edep for Plane {plane}: {scaled_de}")
print(f"Size of scaled_edep_per_plane: {len(scaled_edep_per_plane)}")
# Plot scaled_edep_per_plane vs pathLength (expect a positive correlation)
plt.figure(figsize=(10, 6))
for plane, scaled_de in enumerate(scaled_edep_per_plane):
    plt.scatter(scaled_de, pathLength[a["plane"][hitCut] == plane], alpha=0.5)
plt.xlabel('scaled_edep')
plt.ylabel('pathLength')
plt.title('scaled_edep_per_plane vs pathLength')
plt.grid(True)
plt.show()
# Plot scaled edep values as a histogram
plt.figure(figsize=(10, 6))
for plane, scaled_de in enumerate(scaled_edep_per_plane):
    plt.hist(scaled_de, bins=100, alpha=0.5, label=f'Scaled edep Plane {plane}')
plt.xlabel('Scaled edep')
plt.ylabel('Frequency')
plt.title('Histogram of Scaled edep Values')
plt.legend()
plt.grid(True)
plt.show()
#%% Calculate the ratio of edep between kltsh and kltshmc
dep_ratio = a["edep"][hitCut] / amc["edep"][hitCut]
print(f"Dimensions of dep_ratio: {dep_ratio.shape}")
print(dep_ratio)
# %% Calculate the ratio of scaled edep values to Monte Carlo truth values for each plane
ratio_scaled_edep_per_plane = []
for plane in range(36):
    planeCut = a["plane"][hitCut] == plane
    demc = amc["edep"][hitCut][planeCut]
    scaled_de = scaled_edep_per_plane[plane]
    ratio = scaled_de / demc
    ratio_scaled_edep_per_plane.append(ratio)
    print(f"Ratio of scaled edep to MC truth for Plane {plane}: {ratio}")

#%% Plot the ratio of scaled edep to MC truth values for each plane
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ratio_scaled_edep_per_plane)))
for plane, ratio in enumerate(ratio_scaled_edep_per_plane):
    plt.hist(ratio, bins=100, alpha=0.5, label=f'Plane {plane}', density=True, color=colors[plane])
plt.xlabel('Scaled edep / MC edep')
plt.ylabel('Density')
plt.title('Histogram of Scaled edep / MC edep for Each Plane')
plt.legend()
plt.grid(True)
plt.show()
# find range of scaled edep values that fit to a line with slope of 1
# maybe descale it to get original range of edep values that calibration is valid for
# %%
plt.figure(figsize=(10, 6))
plane = 1  # Selecting the first plane for demonstration
scaled_de = scaled_edep_per_plane[plane]
demc = amc["edep"][hitCut][a["plane"][hitCut] == plane]

# Define the range of scaled edep values for the fit
min_scaled_de = 0  # Example minimum value
max_scaled_de = 0.003  # Example maximum value
filtered_scaled_de = [de for de in scaled_de if min_scaled_de <= de <= max_scaled_de]
filtered_demc = [demc[i] for i, de in enumerate(scaled_de) if min_scaled_de <= de <= max_scaled_de]

# Fit the data to a straight line within the specified range
m, b = np.polyfit(filtered_scaled_de, filtered_demc, 1)
plt.scatter(scaled_de, demc, alpha=0.5, label=f'Plane {plane}')
plt.plot(filtered_scaled_de, m*np.array(filtered_scaled_de) + b, color='red', linewidth=2, label=f'Fit Line (m={m:.2f}, b={b:.2f})')
plt.xlabel('Scaled edep')
plt.ylabel('MC edep')
plt.title('Scatter Plot of Scaled edep vs MC edep for Plane 0 with Fit Line')
plt.legend()
plt.grid(True)
plt.show()
#%% RMS...histogram of scaled edep vs mcedep before and after calibration...how does RMS differ as u change 1.3 etc
# # %%
# plt.figure(figsize=(10, 6))
# plane = 0  # Selecting the first plane for demonstration
# edep = a["edep"][hitCut][a["plane"][hitCut] == plane]
# mcedep = amc["edep"][hitCut][a["plane"][hitCut] == plane]

# # Fit the data to a straight line
# m, b = np.polyfit(edep, mcedep, 1)
# plt.scatter(edep, mcedep, alpha=0.5, label=f'Plane {plane}')
# plt.plot(edep, m*np.array(edep) + b, color='red', linewidth=2, label=f'Fit Line (m={m:.2f}, b={b:.2f})')
# plt.xlabel('Edep')
# plt.ylabel('MC Edep')
# plt.title('Scatter Plot of Edep vs MC Edep for Plane 0 with Fit Line')
# plt.legend()
# plt.grid(True)
# plt.show()
# # %%
# planeCut = a["plane"][hitCut] == 1
# de = a["edep"][hitCut][planeCut]
# scaled_edep_per_plane[0] = [de * (1/1.11)]
# plt.figure(figsize=(10, 6))
# plt.hist((de/amc["edep"][hitCut][planeCut]), bins=200, alpha=0.5, label=f'Plane 0', density=True, color='blue')
# plt.xlabel('Scaled edep / MC edep')
# plt.ylabel('Density')
# plt.title('Histogram of Scaled edep / MC edep for Plane 0')
# plt.legend()
# plt.grid(True)
# plt.show()
# # %%
# planeCut = a["plane"][hitCut] == 0
# pathCut = pathLength >= 1.3
# target_dedx = scaled_edep_per_plane[0][pathCut[planeCut]]/pathLength[planeCut & pathCut]
# mean_target_dedx = np.mean(target_dedx)
# print(mean_target_dedx)
# # %%
# normalized_average_dedx_per_plane = [average_dedx / mean_target_dedx for average_dedx in average_dedx_per_plane]
# scaled_edep_per_plane = []
# for plane in range(36):
#     planeCut = a["plane"][hitCut] == plane
#     de = a["edep"][hitCut]
#     scaled_de = de[planeCut] / normalized_average_dedx_per_plane[plane]
#     scaled_edep_per_plane.append(scaled_de)
#     print(f"Scaled edep for Plane {plane}: {scaled_de}")
# print(f"Size of scaled_edep_per_plane: {len(scaled_edep_per_plane)}")
# # %%
# plt.figure(figsize=(10, 6))
# for plane, scaled_de in enumerate(scaled_edep_per_plane):
#     plt.hist(scaled_de, bins=100, alpha=0.5, label=f'Scaled edep Plane {plane}')
# plt.xlabel('Scaled edep')
# plt.ylabel('Frequency')
# plt.title('Histogram of Scaled edep Values')
# plt.legend()
# plt.grid(True)
# plt.show()
# # %%
# ratio_scaled_edep_per_plane = []
# for plane in range(36):
#     planeCut = a["plane"][hitCut] == plane
#     demc = amc["edep"][hitCut][planeCut]
#     scaled_de = scaled_edep_per_plane[plane]
#     ratio = scaled_de / demc
#     ratio_scaled_edep_per_plane.append(ratio)
#     print(f"Ratio of scaled edep to MC truth for Plane {plane}: {ratio}")

# #%% Plot the ratio of scaled edep to MC truth values for each plane
# plt.figure(figsize=(10, 6))
# ratio = ratio_scaled_edep_per_plane[2]
# plt.hist(ratio, bins=100, alpha=0.5, label='Plane 0', density=True, color='blue')
# plt.xlabel('Scaled edep / MC edep')
# plt.ylabel('Density')
# plt.title('Histogram of Scaled edep / MC edep for Plane 0')
# plt.legend()
# plt.grid(True)
# plt.show()
# # %%
# print(average_dedx_per_plane[1])
# print(average_dedx_per_plane[0])
# # %%
