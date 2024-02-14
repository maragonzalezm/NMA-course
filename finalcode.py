"""
This code was written for brain data analysis, using the Steinmetz et al. (2019) dataset, during a Computational Neuroscience course

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import zscore
from sklearn.decomposition import PCA
from google.colab import files

# @title Figure settings
from matplotlib import rcParams

rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

# @title Data retrieval
import os, requests

fname = []
for j in range(3):
  fname.append('steinmetz_part%d.npz'%j)
url = ["https://osf.io/agvxh/download"]
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")

for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)

# @title Data loading
alldat = np.array([])
for j in range(len(fname)):
  alldat = np.hstack((alldat,
                      np.load('steinmetz_part%d.npz'%j,
                              allow_pickle=True)['dat']))


## This plots a sing neuron at a specific trial, showing the time of stimulus, response and feedback
#dat=alldat[0]
#spikes=dat['spks']
#feedback=dat['feedback_time'][0,0]*100
#resp=dat['response_time'][0,0]*100
#plt.figure
#plt.subplot()
#plt.plot(spikes[1,0,:])
#plt.plot(feedback, 0, marker="o", markersize=20)
#plt.plot(50,0,marker="o", markersize=20)
#plt.plot(resp,0,marker="o",markersize=20)
#plt.show

##This calculates the mean brain activity per region, in 3 different times for each session (baseline, 50ms before response and 50ms after)
sessionmat=[]
for index in range(len(alldat)):
  dat=alldat[index]
  spikes=dat['spks']
  response=[]
  for index in range(len(dat['response_time'])):
    respns=np.round(dat['response_time'][index][0]*100).astype(int)
    response.append(respns)

  listaresp=[]
  for i in range(len(spikes)):
    meanvalue=[]
    for k in range(len(dat['response_time'])):
      val1=np.mean((spikes[i,k,:])[response[k]-50:response[k]])
      meanvalue.append(val1)
      #print(np.mean(spikes[i,k,:]).shape)
      #print(val1.shape,val1)
    listaresp.append(np.sum(meanvalue))

  listastart=[]
  for i in range(len(spikes)):
    val2=np.sum(np.mean(spikes[i,:,:],axis=0)[0:49])
    listastart.append(val2)

  listaafter=[]
  for i in range(len(spikes)):
    val3=np.sum(np.mean(spikes[i,:,:],axis=0)[response[k]:response[k]+50])
    listaafter.append(val3)

  dictreg1={}
  dictreg2={}
  dictreg3={}

  for region in np.unique(dat['brain_area']):
    spkstotal1=0
    spkstotal2=0
    spkstotal3=0
    for k in np.where(dat['brain_area']==region)[0]:
      spkstotal1=spkstotal1+listastart[k]
    dictreg1[region]=spkstotal1/len(np.where(dat['brain_area']==region)[0])

    for j in np.where(dat['brain_area']==region)[0]:
      spkstotal2=spkstotal2+listaresp[j]
    dictreg2[region]=spkstotal2/len(np.where(dat['brain_area']==region)[0])

    for ik in np.where(dat['brain_area']==region)[0]:
      spkstotal3=spkstotal3+listaafter[ik]
    dictreg3[region]=spkstotal3/len(np.where(dat['brain_area']==region)[0])

  dictionary=[dictreg1,dictreg2,dictreg3]

  sessionmat.append(dictionary)

#This obtains the difference in brain activity between the response activity and the baseline
#Note: session 6, 12 and 39 have a NaN result
diff1=[]
diff2=[]
regions=[]
dictionary_final=[]

for i in range(len(sessionmat)):
  valuediff1=np.array(list(sessionmat[i][1].values()))/np.array(list(sessionmat[i][0].values()))
  diff1.append(valuediff1)

  valuediff2=np.array(list(sessionmat[i][2].values()))/np.array(list(sessionmat[i][0].values()))
  diff2.append(valuediff2)

  reg=np.array(list((sessionmat[i][0].keys())))
  regions.append(reg)

#A dictionary is created with the difference value (1 representing no difference, 2 representing double the activity, 0.5 representing half the activity etc) per brain area
dictionary_1=[]
for i in range(len(sessionmat)):
  for k in range(len(regions[i])):
    total=(regions[i][k],diff1[i][k])
    dictionary_1.append(total)

dictionary_2=[]
for i in range(len(sessionmat)):
  for k in range(len(regions[i])):
    total=(regions[i][k],diff2[i][k])
    dictionary_2.append(total)

dict_final1= {}
for reg1, value1 in dictionary_1:
  if reg1 not in dict_final1:
      dict_final1[reg1] = [value1]
  else:
      dict_final1[reg1].append(value1)

dict_final2= {}
for reg2, value2 in dictionary_2:
  if reg2 not in dict_final2:
      dict_final2[reg2] = [value2]
  else:
      dict_final2[reg2].append(value2)

brainregions = ["Visual cortex", "Thalamus", "Hippocampus", "Other cortices", "Midbrain", "Basal ganglia", "Cortical subplate"]
region_colors = ['blue', 'red', 'green', 'purple', 'violet', 'lightblue', 'orange', 'gray']
brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],  # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],  # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP","TT"],  # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"],  # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],  # basal ganglia
                ["BLA", "BMA", "EP", "EPd", "MEA"]  # cortical subplate
                ]

final_values1 = {reg1: np.nanmean(values1) for reg1, values1 in dict_final1.items()}
final_values1 = {key1: value1 for key1, value1 in final_values1.items() if not np.isnan(value1)}
mean_group1 = {region1: np.mean([final_values1.get(key, 0) for key in group1]) for region1, group1 in zip(brainregions, brain_groups)}

final_values2 = {reg2: np.nanmean(values2) for reg2, values2 in dict_final2.items()}
final_values2 = {key2: value2 for key2, value2 in final_values2.items() if not np.isnan(value2)}
mean_group2 = {region2: np.mean([final_values2.get(key, 0) for key in group2]) for region2, group2 in zip(brainregions, brain_groups)}

fig, ax = plt.subplots(figsize=(12, 6))

for group, color in zip(brain_groups, region_colors):
    group_data1 = {key: final_values1[key] for key in group if key in final_values1}
    ax.bar(group_data1.keys(), group_data1.values(), color=color)

ax.set_xlabel('Brain Regions')
ax.set_ylabel('Activity Change in Relation to Basal Activity')
ax.set_title('Brain Regions Activated Before Response')
plt.xticks(rotation=90, size=9.5)
ax.set_ylim(0, 10)

legend = [mpatches.Patch(color=color, label=region) for color, region in zip(region_colors, brainregions)]
plt.legend(handles=legend, loc='upper right', fontsize=8)

plt.savefig('beforeresponse1.png')
#files.download('beforeresponse1.png')
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14, 10))

for group, color in zip(brain_groups, region_colors):
    group_data1 = {key: final_values1[key] for key in group if key in final_values1}
    ax1.bar(group_data1.keys(), group_data1.values(), color=color)

ax1.set_xlabel('Brain Regions')
ax1.set_ylabel('Activity Change in Relation to Baseline Act.')
ax1.set_title('Brain Regions Activated Before Response')
ax1.tick_params(axis='x', rotation=90, labelsize=9.5)
ax1.set_ylim(0, 10)

for group, color in zip(brain_groups, region_colors):
    group_data2 = {key: final_values2[key] for key in group if key in final_values2}
    ax2.bar(group_data2.keys(),group_data2.values(), color=color)

ax2.set_xlabel('Brain Regions')
ax2.set_ylabel('Activity Change in Relation to Baseline Act.')
ax2.set_title('Brain Regions Activated After Response')
ax2.tick_params(axis='x', rotation=90, labelsize=9.5)
ax2.set_ylim(0, 10)


legend = [mpatches.Patch(color=color, label=region) for color, region in zip(region_colors, brainregions)]
plt.legend(handles=legend, loc='upper right', fontsize=8)

plt.savefig('afterresponse1.png')
files.download('afterresponse1.png')
plt.show()

plt.bar(sorted_mean_values1.keys(), sorted_mean_values1.values())
plt.xlabel('Brain Regions')
plt.ylabel('Act. Change in Relation to Basal Act.', size=12.5)
plt.title('Brain Regions Activated Before Response')
plt.xticks(rotation=45, ha='right',size=13)

plt.savefig('beforeresponse2.png')
files.download('beforeresponse2.png')
plt.show()

plt.bar(sorted_mean_values2.keys(), sorted_mean_values2.values())
plt.xlabel('Brain Regions')
plt.ylabel('Act. Change in Relation to Basal Act.', size=12.5)
plt.title('Brain Regions Activated After Response')
plt.xticks(rotation=45, ha='right',size=13)
plt.savefig('afterresponse2.png')
files.download('afterresponse2.png')
plt.show()
