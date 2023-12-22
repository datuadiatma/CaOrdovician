# Load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import  interp1d

plt.rcParams['axes.linewidth']=1.5

# Load data
dx = pd.read_csv('MeiklejohnC.csv')
df = pd.read_csv('MeiklejohnCa.csv')

# Remove samples that do not have d44Ca values, store as df_ca
df_ca = df.dropna(subset=['d44Ca'])
df_srca = df.dropna(subset=['Sr/Ca'])

d13C = dx['d13c'].to_numpy()
d44Ca = df_ca['d44Ca'].to_numpy()
meter_ca = df_ca['Meterage'].to_numpy()

SrCa_data = df_srca['Sr/Ca'].to_numpy()
meter_srca = df_srca['Meterage'].to_numpy()

# Aragonite fractionation (Romanek et al., 1992)
Darg = 2.7

# Calcite fractionation (Romanek et al., 1992)
Dcalc = 1.0

# Aragonite and Calcite mixing model
# ------------------------------------
def mixing(farg, darg=2.7, dcalc=1.0, DIC=0):
    """Isotopic mixing model

    Simplified two endmembers mixing models. This version assumes the elemental
    concentrations in aragonite = calcite.

    Parameters
    ----------
    farg : ndarray
        Aragonite fraction.
    darg : float, optional
        Aragonite fractionation factor. Default is +2.7permil.
    dcalc : float, optional
        Calcite fractionation factor. Default is +1permil.
    DIC : float, optional
        Isotopic ratio of DIC from which aragonite and calcite
        precipitate. Default is 0permil.
    
    Returns
    -------
    d13Ccarb : float
        Isotopic ratio of bulk rock (mixed of aragonite + calcite).
    """
    fcalc = (1 - farg)
    d13Carg = DIC + darg
    d13Ccalc = DIC + dcalc
    d13Ccarb = farg*d13Carg + fcalc*d13Ccalc
    return d13Ccarb

def cmixing(farg, carg, ccalc):
    """Concentration mixing model

    Parameters
    ----------
    farg : ndarray
        Aragonite fraction.
    carg : float 
        Concentration of element in aragonite.
    ccalc : float, optional
        Concentration of element in calcite.
    
    Returns
    -------
    mix : float
        Concentration of element in the mixture / bulk carbonate.
    """
    fcalc = (1 - farg)
    mix = farg*carg + fcalc*ccalc
    return mix

def isomixing(farg, carg, ccalc,darg=2.7, dcalc=1.0, DIC=0):
    """Isotopic mixing model

    Two endmembers mixing models. This is a generalized version of the function
    "mixing".

    Parameters
    ----------
    farg : ndarray
        Aragonite fraction.
    carg : float 
        Concentration of element in aragonite.
    ccalc : float, optional
        Concentration of element in calcite.
    darg : float, optional
        Aragonite fractionation factor. Default is +2.7permil.
    dcalc : float, optional
        Calcite fractionation factor. Default is +1permil.
    DIC : float, optional
        Isotopic ratio of DIC from which aragonite and calcite
        precipitate. Default is 0permil.
    
    Returns
    -------
    d13Ccarb : float
        Isotopic ratio of bulk rock (mixed of aragonite + calcite).
    """
    cmix = cmixing(farg, carg, ccalc)
    fcalc = (1 - farg)
    d13Carg = DIC + darg
    d13Ccalc = DIC + dcalc
    d13Ccarb = farg*(carg/cmix)*d13Carg + fcalc*(ccalc/cmix)*d13Ccalc
    return d13Ccarb

# ------------------------------------

farg = np.linspace(0, 1, 5)
# Concentration in ppm
Ca_calcite = 400551
Sr_calcite = 500

# Concentration in ppm
Ca_aragonite = 397120
Sr_aragonite = 8000

# Concentration in percent
C_calcite = 43.97
C_aragonite = 43.97

Calccarb = cmixing(farg, Ca_aragonite, Ca_calcite)
Srcarb = cmixing(farg, Sr_aragonite, Sr_calcite)

# Convert to Sr/Ca in mmol/mol
SrCa = Srcarb / Calccarb * (40.078/97.62) * 1000

isocarb = isomixing(farg, C_aragonite, C_calcite, DIC=-3.7)
isoCacarb = isomixing(farg, C_aragonite, C_calcite, darg=-1.5, dcalc=-0.9)

def subplots_centered(nrows, ncols, figsize, nfigs):
    """Modification of matplotlib plt.subplots().
    
    This function is useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

# Generate mineral fraction curves from mixing model
fca = interp1d(isoCacarb, farg, fill_value='extrapolate')
fsrca = interp1d(SrCa, farg, bounds_error=False, fill_value='extrapolate')

# Aragonite and Calcite fraction from Ca isotope
frac_arag_ca = fca(d44Ca)
frac_calc_ca = 1 - frac_arag_ca

# Aragonite and Calcite fraction from Sr/Ca
frac_arag_srca = fsrca(SrCa_data)
frac_calc_srca = 1 - frac_arag_ca
  
# Plotting
# ---------------------------------
fig, axs = subplots_centered(nrows=2, ncols=2, figsize=(10,8), nfigs=3)

ax = axs[0]
ax1 = axs[1]
ax2 = axs[2]

# Random number generator with 614 as seed for reproducibility
rng = np.random.default_rng(614)

# Generate random x positions around 1.1 to plot data
xdum1 = rng.normal(1.1, 0.01, len(d13C))

fca = interp1d(isoCacarb, farg)
fargca = fca(np.mean(d44Ca))
# Generate random x positions around mean d44Ca to plot data
xdum2 = rng.normal(fargca, 0.01, len(d44Ca))

fsrca = interp1d(SrCa, farg)
fargsrca = fsrca(np.mean(SrCa_data))
# Generate random x positions around mean Sr/Ca to plot data
xdum3 = rng.normal(fargsrca, 0.01, len(SrCa_data))

ax.plot(farg, isocarb)
ax.scatter(farg, isocarb, fc='k', marker='s', zorder=5)
ax.scatter(xdum1, d13C, fc='steelblue', ec='grey', alpha=0.3, zorder=6)
ax.axhline(y=-3.7, ls='--', c='k')
for i, j in zip(farg,isocarb):
    dx, dy = -0.04, -0.4
    ax.text(i+dx, j+dy, '{:.0f}%'.format(i*100), fontsize=12)

ax.text(0, -3.6, '$\delta^{13}C_{DIC}$', fontsize=12)
ax.set_xlabel('$f_{aragonite}$', fontsize=14)
ax.set_ylabel('$\delta^{13}C_{carbonate}$', fontsize=14)
# ax.set_ylim(-0.4, 2.9)
ax.set_xlim(-0.1, 1.15)

axt = ax.twiny()
axt.set_xlim(1-(-0.1), 1-1.15)
axt.set_xlabel('$f_{calcite}$', fontsize=14)

# --------------
ax1.plot(farg, isoCacarb, c='magenta')
ax1.scatter(farg, isoCacarb, fc='k', marker='s', zorder=5)
ax1.scatter(xdum2, d44Ca, fc='magenta', ec='grey', alpha=0.3, zorder=6)
ax1.axhline(y=0, ls='--', c='k')
for i, j in zip(farg,isoCacarb):
    dx, dy = -0.04, -0.2
    ax1.text(i+dx, j+dy, '{:.0f}%'.format(i*100), fontsize=12)

ax1.text(0, 0.05, '$\delta^{44/40}Ca_{seawater}$', fontsize=12)
ax1.set_xlabel('$f_{aragonite}$', fontsize=14)
ax1.set_ylabel('$\delta^{44/40}Ca_{carbonate}$', fontsize=14)
ax1.set_ylim(-1.8, 0.2)
ax1.set_xlim(-0.1, 1.15)
ax1.tick_params(axis='y', left=False, right=True, labelleft=False,
                labelright=True)
ax1.yaxis.set_label_position('right')

axt1 = ax1.twiny()
axt1.set_xlim(1-(-0.1), 1-1.15)
axt1.set_xlabel('$f_{calcite}$', fontsize=14)

#-----------------
ax2.plot(farg, SrCa, c='orange')
ax2.scatter(farg, SrCa, fc='k', marker='s', zorder=5)
ax2.scatter(xdum3, SrCa_data, fc='orange', ec='grey', alpha=0.3, zorder=6)

for i, j in zip(farg,SrCa):
    dx, dy = 0.02, -0.5
    ax2.text(i+dx, j+dy, '{:.0f}%'.format(i*100), fontsize=12)


ax2.set_xlabel('$f_{aragonite}$', fontsize=14)
ax2.set_ylabel('$Sr/Ca$', fontsize=14)

ax2.set_xlim(-0.1, 1.15)

axt2 = ax2.twiny()
axt2.set_xlim(1-(-0.1), 1-1.15)
axt2.set_xlabel('$f_{calcite}$', fontsize=14)


for axo in [ax, ax1, ax2]:
    axo.grid(which='both', linestyle=':')
    axo.yaxis.set_major_locator(plt.MaxNLocator(5))

ax.scatter(1.1, np.mean(d13C), s=150, marker='s',fc='steelblue', ec='k', zorder=5)
ax.errorbar(1.1, np.mean(d13C), yerr=2*np.std(d13C), c='black', zorder=4)

ax1.scatter(fargca, np.mean(d44Ca), s=150, marker='s',fc='magenta', ec='k', zorder=5)
ax1.errorbar(fargca, np.mean(d44Ca), yerr=2*np.std(d44Ca), c='black', zorder=4)

ax2.scatter(fargsrca, np.mean(SrCa_data), s=150, marker='s',fc='orange', ec='k', zorder=5)
ax2.errorbar(fargsrca, np.mean(SrCa_data), yerr=2*np.std(SrCa_data), c='black', zorder=4)


# Mineral Curves plot
fig2, ((axc1, axc2), (axc3, axc4))  = plt.subplots(2, 2, figsize=(7, 11), sharey=True)

axc1.set_ylim(meter_ca.min(), meter_ca.max())

axc1.scatter(d44Ca, meter_ca, ec='k', fc='magenta', s=80)
axc2.plot(frac_arag_ca, meter_ca, 'k-')
axc2.fill_betweenx(meter_ca, x1=0, x2=frac_arag_ca, label='Aragonite', 
                  fc='steelblue', hatch='/')
axc2.fill_betweenx(meter_ca, x1=frac_arag_ca, x2=1, label='Calcite',
                  fc='lightgrey', hatch='|')
axc2.legend()

axc3.scatter(SrCa_data, meter_srca, ec='k', fc='orange', s=80)
axc4.plot(frac_arag_srca, meter_srca, 'k-')
axc4.fill_betweenx(meter_srca, x1=0, x2=frac_arag_srca, label='Aragonite', 
                  fc='steelblue', hatch='/')
axc4.fill_betweenx(meter_srca, x1=frac_arag_srca, x2=1, label='Calcite',
                  fc='lightgrey', hatch='|')
axc4.legend()

axc1.set_xlabel('$\delta^{44/40}Ca_{carbonate}$', fontsize=14)
axc3.set_xlabel('Sr/Ca', fontsize=14)

for ax in (axc1, axc3):
    ax.set_ylabel('Meter', fontsize=14)
    ax.grid()

for ax in (axc2, axc4):
    ax.set_xlim(0,1)
    ax.set_xlabel('$f_{mineral}$', fontsize=14)

plt.tight_layout()
plt.show()