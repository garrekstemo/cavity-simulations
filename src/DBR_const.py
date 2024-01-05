import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
from matplotlib import animation
from meep.materials import Au
from meep.materials import Ge

datetime = datetime.now().strftime("%Y%m%d_%H%M")

resolution = 32
frequency = 0.22
frequencyWidth = 0.15
numberFrequencies = 1000
pmlThickness = 2.0
animationTimestepDuration = 0.2
powerDecayTarget = 1e-9

n0 = 1.0	# Air
t1 = 0.51512	# ZnS
t2 = 0.29066	# Ge
n1 = 2.25789
n2 = 3.97531
ns = 1.401	# CaF2
ts = 1
tcav=12	# cavity thickness

endTime = 3000; #400

###########
#define ZnS material
# wavelength range: 0.4-14μm: Real only

um_scale=1.0
ZnS_range = mp.FreqRange(min=um_scale/14, max=um_scale/0.4)

ZnS_frq1 = 6.530
ZnS_gam1 = 0
ZnS_sig1 = 3.619
ZnS_frq2 = 3.163
ZnS_gam2 = 0
ZnS_sig2 = 0.508
ZnS_frq3 = 0.0295
ZnS_gam3 = 0
ZnS_sig3 = 2.220

# 電気感受率を求める
ZnS_susc = [mp.LorentzianSusceptibility(frequency=ZnS_frq1, gamma=ZnS_gam1, sigma=ZnS_sig1),
           	mp.LorentzianSusceptibility(frequency=ZnS_frq2, gamma=ZnS_gam2, sigma=ZnS_sig2),
	mp.LorentzianSusceptibility(frequency=ZnS_frq3, gamma=ZnS_gam3, sigma=ZnS_sig3)]

ZnS = mp.Medium(epsilon=1.010356, E_susceptibilities=ZnS_susc, valid_freq_range=ZnS_range)
###########
# define CaF2 material
# wavelength range: 0.23-9.7μm: Real only

CaF2_range = mp.FreqRange(min=um_scale/9.7, max=um_scale/0.23)

CaF2_frq1 = 19.895
CaF2_gam1 = 0
CaF2_sig1 = 0.5676
CaF2_frq2 = 9.9612
CaF2_gam2 = 0
CaF2_sig2 = 0.4711
CaF2_frq3 = 0.0289
CaF2_gam3 = 0
CaF2_sig3 = 3.8485

CaF2_susc = [mp.LorentzianSusceptibility(frequency= CaF2_frq1, gamma= CaF2_gam1, sigma= CaF2_sig1),
           	mp.LorentzianSusceptibility(frequency= CaF2_frq2, gamma= CaF2_gam2, sigma= CaF2_sig2),
	mp.LorentzianSusceptibility(frequency= CaF2_frq3, gamma= CaF2_gam3, sigma= CaF2_sig3)]

CaF2 = mp.Medium(epsilon=1.0, E_susceptibilities= CaF2_susc, valid_freq_range= CaF2_range)
###########

layerIndexes = np.array([n0, ns, n2, n1, n2, n1, n2, n1, n2, n1, n0, n1, n2, n1, n2, n1, n2, n1, n2, ns, n0])	#total 21 layers
layerThicknesses = np.array([50, ts, t2, t1, t2, t1, t2, t1, t2, t1, tcav, t1, t2, t1, t2, t1, t2, t1, t2, ts, 50])
layerThicknesses[0] += pmlThickness
layerThicknesses[-1] += pmlThickness
length = np.sum(layerThicknesses)
layerCenters = np.cumsum(layerThicknesses) - layerThicknesses/2
layerCenters = layerCenters - length/2

sourceLocation = mp.Vector3(0, 0, layerCenters[0] - layerThicknesses[0]/4)
transmissionMonitorLocation = mp.Vector3(0, 0, layerCenters[-1]-pmlThickness/2)
reflectionMonitorLocation = mp.Vector3(0, 0, layerCenters[0] + layerThicknesses[0]/4)


# 角度をつける
theta = np.radians(30)
minimumFrequency = frequency - frequencyWidth/2
kVector = mp.Vector3(np.sin(theta), 0, np.cos(theta)).scale(minimumFrequency)

cellSize = mp.Vector3(0, 0, length)

# file_skip=open('12-4(DBR)_Efield.txt','w')
# writer_skip=csv.writer(file_skip)

sources = [mp.Source(
    mp.GaussianSource(frequency=frequency, fwidth=frequencyWidth),
    component=mp.Ex,
    center=sourceLocation)  ]

    
geometry = [mp.Block(mp.Vector3(mp.inf, mp.inf, layerThicknesses[i]),
    center=mp.Vector3(0, 0, layerCenters[i]), material=mp.Medium(index=layerIndexes[i]))
    for i in range(layerThicknesses.size)]
    
        
pmlLayers = [mp.PML(thickness=pmlThickness)]

simulation = mp.Simulation(
    cell_size=cellSize,
    sources=sources,
    resolution=resolution,
    dimensions=1,
    boundary_layers=pmlLayers,
    k_point=kVector)

incidentRegion = mp.FluxRegion(center=reflectionMonitorLocation)
incidentFluxMonitor = simulation.add_flux(frequency, frequencyWidth, numberFrequencies, incidentRegion)

simulation.run(until_after_sources=endTime)

incidentFluxToSubtract = simulation.get_flux_data(incidentFluxMonitor)

# シミュレーションから電場Exの成分を取り出す
# centerはExを取得する領域の中心、componentで取得するフィールドの成分を指定、sizeは取得する領域のサイズ
fieldEx = np.real(simulation.get_array(center=mp.Vector3(0, 0, 0), size=cellSize, component=mp.Ex))

# fieldExからデータを格納するfieldDataという箱を作る。
fieldData = np.zeros(len(fieldEx))

simulation = mp.Simulation(
    cell_size=cellSize,
    sources=sources,
    resolution=resolution,
    boundary_layers=pmlLayers,
    dimensions=1,
    geometry=geometry,
    k_point=kVector)

transmissionRegion = mp.FluxRegion(center=transmissionMonitorLocation)
transmissionFluxMonitor = simulation.add_flux(frequency, frequencyWidth, numberFrequencies, transmissionRegion)
reflectionRegion = incidentRegion
reflectionFluxMonitor = simulation.add_flux(frequency, frequencyWidth, numberFrequencies, reflectionRegion)
simulation.load_minus_flux_data(reflectionFluxMonitor, incidentFluxToSubtract)


# 'Refl12_4(DBR)_R_and_T.txt'というファイルを新規に作成し、その先頭に"freq", "wavelength", "R", "T"のヘッダーを持つ行を書き込んでいる   
# with open('Refl12_4(DBR)_R_and_T.txt','w',newline="") as f:
#     wave_label=csv.writer(f,delimiter="\t")
#     wave_label.writerow(["freq","wavelength","R","T"])

# fieldDataに電場のデータを格納する
def updateField(sim):
    global fieldData
    fieldEx = np.real(sim.get_array(center=mp.Vector3(0, 0, 0), size=cellSize, component=mp.Ex))
    # fieldExSkip = fieldEx.copy()
    # fieldExSkip = fieldExSkip[::1000]
    
    fieldData = np.vstack((fieldData, fieldEx))
    # writer_skip.writerow(fieldEx)
    

simulation.run(mp.after_sources(mp.Harminv(mp.Ex, mp.Vector3(0, 0, layerCenters[2]), frequency, frequencyWidth)),
        until_after_sources=endTime)
#simulation.run(mp.after_sources(mp.Harminv(mp.Ex, mp.Vector3(0, 0, layerCenters[2]), frequency, frequencyWidth)),
#        mp.at_every(animationTimestepDuration, updateField),
#        until_after_sources=endTime)


frequencies = np.array(mp.get_flux_freqs(reflectionFluxMonitor))

incidentFlux = np.array(mp.get_fluxes(incidentFluxMonitor))
transmittedFlux = np.array(mp.get_fluxes(transmissionFluxMonitor))
reflectedFlux = np.array(mp.get_fluxes(reflectionFluxMonitor))
R = -reflectedFlux / incidentFlux
T = transmittedFlux / incidentFlux
print(R + T)

frequencies = np.array(mp.get_flux_freqs(reflectionFluxMonitor))

Data1D_RT = []
Data1D_RT.append(frequencies) #appendでData1D_RTに格納
Data1D_RT.append(1/frequencies)
Data1D_RT.append(R)
Data1D_RT.append(T)
array1=np.array(Data1D_RT)
# with open('Refl12_4(DBR)_R_and_T.txt','a') as f_handle:
#         np.savetxt(f_handle, array1.transpose())


# "-----------テキストファイルを作成する-------------"

# textfolderpath = "text"
# textfilename = "231208_01.txt"
# textfilepath = os.path.join(textfolderpath, textfilename)
# with open(textfilepath,'a') as f_handle:
#         np.savetxt(f_handle, array1.transpose())

"------------スペクトルをプロット、保存-----------"
reflectionSpectraFigure = plt.figure()
reflectionSpectraAxes = plt.axes(xlim=(frequency-0.01-frequencyWidth/2, frequency+0.01+frequencyWidth/2),ylim=(0, 1.2))
reflectionLine, = reflectionSpectraAxes.plot(frequencies, R, lw=2)
reflectionSpectraAxes.set_xlabel('frequency (um / \u03BB0)')
reflectionSpectraAxes.set_ylabel('R')
# reflectionSpectraFigure.savefig("R_spec_12-2(Au).jpg")

# plt.savefig(figfilepath)
plt.show()


(x, y, z, w) = simulation.get_array_metadata()

indexArray = np.sqrt(simulation.get_epsilon())

fig = plt.figure()
ax = plt.axes(xlim=(min(z),max(z)),ylim=(-1,1))
line, = ax.plot([], [], lw=2)




