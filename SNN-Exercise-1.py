#!/usr/bin/env python
# coding: utf-8

# # 1 Introduction to Spiking Neural Networks (SNN) – D7046E @ LTU.SE
# 
# The figure below illustrates a single neuron, see [Figure 1.2A](https://neuronaldynamics.epfl.ch/online/Ch1.S1.html) in Neuronal Dynamics. The dendrite, soma, and axon can be clearly distinguished.
# 
# ![Neuron](https://neuronaldynamics.epfl.ch/online/x2.png)
# 
# Neurons communicate via action potentials that propagate along axons (nerves) to synapses on the dendrites of the target/receiving neurons. The most important feature of an action potential is the **time** when it occurs, while their amplitudes and shapes are similar. In spiking neuron models, action potentials are represented by discrete events called **spikes** that are defined by the time of occurrence and some kind of identifier of the source neuron.
# 
# Spiking neuron models mimic natural neurons more closely than the activation-function based *units* of artificial neural networks (ANNs), which operate with *amplitudes* that qualitatively correspond to average spike frequencies. Spiking neurons have time-dependent states and can for example respond differently to two inputs with identical amplitudes but different temporal structure. Spiking neuron models are central in neuroscience and [neuromorphic engineering](https://www.frontiersin.org/journals/neuroscience/sections/neuromorphic-engineering).
# 
# In this exercise you will investigate and perform numerical experiments with the leaky integrate-and-fire (LIF) model of neurons, see [Figure 1.6A](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) in the book and the lectures for further information.
# 
# ![LIF model](https://neuronaldynamics.epfl.ch/online/x12.png)
# <!-- ![LIF model](https://lcnwww.epfl.ch/gerstner/SPNM/img378.gif) -->
# 
# In the LIF model we consider the cell membrane (big circle) of the neuron as a capacitor, $C$, that integrates an input current, $I(t)$, of ions from synapses. In addition, there is a parallel branch with a voltage source, $u_{rest}$, and a resistor, $R$, which corresponds to ion pumps and leakage of the cell membrane, respectively. When the membrane voltage, $u(t)$, reaches a threshold voltage, $u_{th}$, a spike is generated and $u(t)$ is reset. See the book for details.
# 
# Simulators like [Brian2](https://brian2.readthedocs.io) and [Neuronify](https://ovilab.net/neuronify/) can be used to quickly configure and simulate spiking neural networks (SNNs) of LIF neurons. However, in order to develop an understanding of spiking neurons and neural networks you will solve the corresponding differential equations numerically in this exercise. Some sample code is included below to get you started, and the resulting code will be reused in a subsequent SNN exercise. 
# 
# You will also investigate the delta modulator receptor neuron model introduced in the lectures, which introduces a central concept used in neuromorphic sensor systems like [dynamic vision sensors](https://inivation.com/dvp/). The receptor neuron model can be used to convert input signals to spikes that can be further processed with an SNN.
# 
# There are 10 mandatory tasks and a few optional tasks that you can perform to deepen your knowledge.

# ## 1.1 Libraries and generic functions

# In[2]:


# Enable inline plots in the notebook
# get_ipython().run_line_magic('matplotlib', 'inline')

# Import library functions needed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set default figure size
plt.rcParams['figure.figsize'] = [10, 10]

# Function that is used to plot spike times
def rasterplot(ax, x, y, x_label, y_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y, marker='|')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# ## 1.2 LIF neuron model
# 
# The LIF model is defined by [Equation 1.5](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) in the book $$\tau_m \frac{du}{dt} = -(u - u_{rest}) + R I_{syn},$$ where $u=u(t)$ is the membrane potential, $R$ is the membrane resistance and $\tau_m$ is the membrane time constant, which is related to the time constant of the RC electric circuit in the illustration above. Note that $u(t)$ is a voltage difference across the neuron cell membrane.
# 
# The equilibrium potential $u_{rest}\simeq-65$ mV is determined by the dynamic equilibrium between the ion flow through  channels in the membrane (leakage) and the active ion transport by ion pumps (that maintain a concentration difference). Inside the cell the concentration of ions is different from that in the surrounding liquid. The difference in concentration generates an equilibrium electrical potential called the *Nernst potential* which plays an important role in neuronal dynamics, see [Section 2.1](https://neuronaldynamics.epfl.ch/online/Ch2.S1.html).
# 
# Use a threshold membrane potential of $u_{\text th}=-50$ mV and a reset potential of $u_{reset}=u_{rest}$ in the following tasks, unless stated otherwise. Use a membrane time constant of $\tau_m=30$ ms and a membrane resistance of $R=95$ M$\Omega$.
# 
# $I_{syn}$ is the total input current to the neuron, which is the sum of postsynaptic currents from all synapses on the neuron. This current can also be defined as a constant injection current to investigate the spike frequency of the LIF neuron for different input currents. This is what you will do in this first task.

# **Task 1:** Complete the code below and calculate the membrane potential $u(t)$ for an input current of $I_{syn}=50$ pA. Plot the membrane potential in the interval $t\in[0,1]$ s. Use the Euler forward method to integrate the differential equation with a timestep of $dt=10^{-5}$ s.
# 
# **Check your solution:** You can calculate the asymptotic value of u(t) after long time by setting $du/dt=0$ in the differential equation above. This way you can verify that the numerical solution converges to the correct value – which is somewhere between *-61 and -60 mV*.

# In[3]:


# Parameters for Tasks 1–5

# Timestep
dt = 10**(-5)

# Neuron parameters
R       = 95 * 10**6 
tau_m   = 30 * 10**(-3)
u_rest  = -65 * 10**(-3) 
u_reset = -65 * 10**(-3)
u_thres = -50 * 10**(-3)


# In[4]:


# Code for Task 1

# Input current from synapses
I_syn = 50 * 10**(-12)

# Neuron identifier, only one neuron in this simulation
n_id    = 0

# Placeholders for u(t), used for plotting
t_i = []
u_i = []

# Placeholders for list of spike times and neuron ID's
t_spike = []
n_spike = []

# Simulate one second of time
t = 0
u = u_rest
while t <= 1:
    
    # Euler forward step
    du_dt = (-(u - u_rest) + R * I_syn) / tau_m 
    u += du_dt * dt
    u_i.append(u)            # Store u(t)

    # Spike condition
    if u >= u_thres :
        t_spike.append(t)    # spike time
        n_spike.append(n_id) # neuron ID
        u = u_reset
    
    # Timestep completed
    t_i.append(t)
    t += dt
    
# Plot u(t)    
fig, ax = plt.subplots()
ax.plot(t_i,u_i)
ax.set(xlabel='Time [s]', ylabel='Membrane potential u(t)')
ax.grid()


# **Task 2:** Increase the input current $I_{syn}$ and repeat the simulation. At what integer value of the input current $I_{syn}$ in pA does the neuron start spiking? Plot the membrane potential $u(t)$ for this integer value of $I_{syn}$.
#     
# **Answer:** ??? pA
# 
# **Check your solution:** The expression that you derived in the first task for the asymptotic value of $u(t)$ can be used to estimate the current $I_{syn}$ required to reach an asymptotic value of $u_{thres}$. The correct value is between *150 and 160 pA*.

# In[5]:


# Code for Task 2 


t_spike = []
I_syn = 50 * 10**(-12)
while t_spike == []:
    # Input current from synapses
    n_id    = 0
    t_i = []
    u_i = []
#     t_spike = []
    n_spike = []
    # Simulate one second of time
    t = 0
    u = u_rest
    while t <= 1:
        # Euler forward step
        du_dt = (-(u - u_rest) + R * I_syn) / tau_m 
        u += du_dt * dt
        u_i.append(u)            # Store u(t)
        # Spike condition
        if u >= u_thres :
            t_spike.append(t)    # spike time
            n_spike.append(n_id) # neuron ID
            u = u_reset
        # Timestep completed
        t_i.append(t)
        t += dt
    I_syn += 1* 10**(-12)
print("The correct value is ", I_syn/( 10**(-12)), " pA")
# Plot u(t)    
fig, ax = plt.subplots()
ax.plot(t_i,u_i)
ax.set(xlabel='Time [s]', ylabel='Membrane potential u(t)')
ax.grid()
    


# **Task 3:** Calculate the number of spikes generated during 1 second of simulation time for the input currents 200 pA, 400 pA and 600 pA, respectively. For each value of $I_{syn}$ plot the membrane potential $u(t)$ and the spikes generated using the code provided below.
# 
# **Answers:**
# 
# * With $I_{syn}=200$ pA there are ??? spikes generated in one second
# * With $I_{syn}=400$ pA there are ??? spikes generated in one second
# * With $I_{syn}=600$ pA there are ??? spikes generated in one second
# 
# **Check your solution:** For an input current of 250 pA you should get 33 spikes in one second of simulation time.

# In[6]:


# Code for Task 3

currents = [200, 250, 400, 600]

for i in currents:
    I_syn = i * 10**(-12)
    t_spike = []
    while t_spike == []:
        # Input current from synapses
        n_id    = 0
        t_i = []
        u_i = []
    #     t_spike = []
        n_spike = []
        # Simulate one second of time
        t = 0
        u = u_rest
        while t <= 1:
            # Euler forward step
            du_dt = (-(u - u_rest) + R * I_syn) / tau_m 
            u += du_dt * dt
            u_i.append(u)            # Store u(t)
            # Spike condition
            if u >= u_thres :
                t_spike.append(t)    # spike time
                n_spike.append(n_id) # neuron ID
                u = u_reset
            # Timestep completed
            t_i.append(t)
            t += dt

    print("During one second of time %d spikes were generated for an input current of %d pA" % (len(t_spike),I_syn/1e-12))

    fig,(ax1,ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(t_i,u_i); ax1.set_ylabel('Membrane potential u(t)')
    rasterplot(ax2, t_spike, n_spike,'Time [s]','Spiking neuron ID')


# **Task 4:** Plot the spike frequency in Hz for varying input currents in the range $I_{syn}\in[0, 600]$ pA. This type of plot is called an *f-I Curve* because it illustrates the relation between average spike frequency (f) and stimulus current (I).
# 
# **Check your solution:** Compare the result with the answers in Task 3.
# 
# **Improve your understanding:** The resulting plot should look similar to the ReLU activation function used in ANNs. Compare your plot with an illustration of the ReLU activation function. What does the horizontal axis represent in the two cases? What does the vertical axis represent in the two cases?

# ### 1.2.1 Pulse input
# 
# When a synapse is stimulated by one presynaptic spike a current pulse is generated, see [Figure 3.2](https://neuronaldynamics.epfl.ch/online/Ch3.S1.html) for examples. The amplitude of the current pulse can be a few hundred pA and depends on the efficacy (weight) of the synapse. $I_{syn}$ represents the sum of current pulses from all synapses on the neuron.
# 
# In the next task you will investigate the effect on the membrane potential $u(t)$ of one current pulse of varying amplitude and duration, see [Figures 1.6B and 1.7](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html).

# In[8]:


# Code for Task 4

I_syn_i = []
freq_i = []
for i in range(0,601,1):
    I_syn = i*10**(-12)
    n_id  = 0

        # Placeholders for u(t), used for plotting
    t_i = []
    u_i = []

        # Placeholders for list of spike times and neuron ID's
    t_spike = []
    n_spike = []
    # Simulate one second of time
    t = 0
    u = u_rest

    while t <= 1:

            # Euler forward step
        du_dt = (-(u - u_rest) + R * I_syn) / tau_m 
        u += du_dt * dt
        u_i.append(u)            # Store u(t)

            # Spike condition
        if u >= u_thres :
            t_spike.append(t)    # spike time
            n_spike.append(n_id) # neuron ID
            u = u_reset
            # freq = round(1/np.mean([t_spike[i+1]-t_spike[i]for i in range(len(t_spike)-1)]),2)

            # Timestep completed
        t_i.append(t)
        t += dt
    freq = len(t_spike)
    freq_i.append(freq)
    I_syn_i.append(I_syn*(10**12))


# Plot current vs frequency
fig, ax = plt.subplots()
ax.plot(I_syn_i, freq_i)
ax.set(xlabel='I_syn [pA]', ylabel='Spike rate [Hz]')
ax.grid()




# **Task 5:** Repeat the simulation with the input current $I_{syn}$ set to a constant value in the time interval $t\in[0,t_{stimuli}]$ and zero otherwise, where $t_{stimuli}$ is an integer number of milliseconds. At what minimum value of $t_{stimuli}$ does the neuron fire one spike for $I_{syn} \in [200, 400, 600]$ pA? Plot the membrane potential and be prepared to explain the relation between [Figure 1.7](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) in the book and your results.
# 
# **Answer:**
# 
# For $I_{syn}=200$ pA the neuron fires one spike when $t_{stimuli}$ = ??? milliseconds.
# 
# For $I_{syn}=400$ pA ???
# 
# For $I_{syn}=600$ pA ???
# 
# **Check your solution:** For $I_{syn}=400$ pA, the minimum value of $t_{stimuli}$ for triggering a spike is somwhere between 10 and 20 ms.

# In[11]:


# Code for Task 5

t_st = []

currents = [200,400,600]
for index, i in enumerate(currents):
    I_syn = i*10**(-12) 
    n_spike = []
    t_stimuli = 0

    while not len(n_spike):
        t_stimuli += 1e-1

        n_id    = 0

        # Placeholders for u(t), used for plotting
        t_i = []
        u_i = []

        # Placeholders for list of spike times and neuron ID's
        t_spike = []

        # Simulate one second of time
        t = 0
        u = u_rest
    
        while t <= 0.15:
            if t >= t_stimuli:
                I_syn = 0
            # Euler forward step
            du_dt = (-(u - u_rest) + R * I_syn) / tau_m 
            u += du_dt * dt
            u_i.append(u)            # Store u(t)

            # Spike condition
            if u >= u_thres :
                t_spike.append(t)    # spike time
                n_spike.append(n_id) # neuron ID
                u = u_reset
                if len(t_st) < index+1 :
                    t_st.append(t)

                    print("For Isyn = ", i, " pA the neuron fires one spike when tstimuli = ", t*1000," ms")

    #             break
            # Timestep completed
            t_i.append(t)
            t += dt

        ax.plot(t_i, u_i)

fig, ax = plt.subplots()
ax.set(xlabel='Time [s]', ylabel='Membrane potential u(t)')
ax.grid()


