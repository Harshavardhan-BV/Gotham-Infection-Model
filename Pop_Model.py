#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from multiprocessing import Pool, current_process
import seaborn as sns
import matplotlib.pyplot as plt
#%%
## Input parameters
# No of people in the population
n_agent = 1000
# Parameters
# Clearance rate
cr_mu = 306
cr_sigma = 0.1*cr_mu
# Intercompartmental rate
icr_mu = 2611.8
icr_sigma = 0.1*icr_mu
# Central volume
cv_mu = 3138
cv_sigma = 0.1*cv_mu
# Peripheral volume
pv_mu = 2718
pv_sigma = 0.1*pv_mu
# Dose to be safe
min_dose = 5*1e-3
# Days between doses
t_max = 240 
#%%
# Define ODE for PK model
def PK_model(t, y, a):
    dose_cent, dose_peri = y
    cr, icr, cv, pv = a
    # Equations are wrong need to change
    dy1 = (- dose_cent*cr - dose_cent*icr + dose_peri*icr)/cv
    dy2 = (icr*dose_cent - icr*dose_peri)/pv
    return np.array([dy1, dy2])
#%%
# Wrapper to solve ODE for each agent
def sol_wrap(a):
    *a, dose = a
    return solve_ivp(PK_model, (0, t_max), [dose, 0], args=(a,), dense_output=True)
#%%
# Define Force of Infection
def FOI(t):
    # yearly seasonality
    t = t % 365
    # Given in problem statement
    BL = 2.25
    A = 2633
    sig = 48
    mu = 162
    return BL + A * np.exp(-((t - mu)**2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))
#%% 
# Probability of infection
def P_inf(t):
    # Assuming a constant per capita rate of infection within dt
    return 1-np.exp(-FOI(t)/n_agent)
#%%
def infect(dose, infected, t):
    if infected:
        # If infected stay infected
        return 1
    elif dose >= min_dose:
        # If dose high stay uninfected
        return 0
    else:
        # get random number
        r = np.random.random()
        # If random number < FOI, become infected
        if r < P_inf(t):
            return 1
        else:
            return 0
#%%
def simulation(dose, t_d):
    # Gothamite given as a array of parameters clearance rate, intercompartmental rate, central volume, peripheral volume
    cr = np.random.normal(cr_mu, cr_sigma, n_agent)
    icr = np.random.normal(icr_mu, icr_sigma, n_agent)
    cv = np.random.normal(cv_mu, cv_sigma, n_agent)
    pv = np.random.normal(pv_mu, pv_sigma, n_agent)
    dosage = np.full(n_agent, dose)
    gothamite_param = np.array([cr, icr, cv, pv,dosage]).T
    # Make n_agent solution objects for the ODE 
    gothamite_sol = [sol_wrap(a) for a in gothamite_param]
    # All agents start uninfected
    gothamite_infected = np.zeros(n_agent)
    # Run simulation only until next dose
    for t_a in range(0, t_max):
        # Shifting t by t_d
        t = t_a + t_d
        # Update the dose of each agent by solving the ODE
        dosage = [soli.sol(t_a)[0] for soli in gothamite_sol]
        # Run infection probability check
        gothamite_infected = np.vectorize(infect)(dosage, gothamite_infected, t)
    # get the number infected 
    return gothamite_infected.sum()
# %%
def simulate_replicates(dose, t_d, n_rep):
    # Run n_rep simulations 
    return np.array([simulation(dose, t_d) for _ in range(n_rep)])
# %%
# Find optimal dose by running simulation for different doses and time of dose
params = [(dose, t_d, 10) for dose in range(5,50,5) for t_d in range(0, 365, 30)]
with Pool() as pool:
    viability = pool.starmap(simulate_replicates, params)
#%%
# Convert to dataframe and save output
df = pd.DataFrame(params, columns=['dose', 't_d', 'n_rep'])
df['mean_infected'] = np.array(viability).mean(axis=1)
df['viability'] = (np.array(viability)<750).all(axis=1)
df.to_csv('./outputs/viability.csv', index=False)
# %%
# Plot heatmap of viability
inf = df.pivot(index='dose', columns='t_d', values='mean_infected')
annot = df.pivot(index='dose', columns='t_d', values='viability')
sns.heatmap(inf, annot=annot ,cmap='viridis')
plt.savefig('./outputs/viability.pdf')
# %%
