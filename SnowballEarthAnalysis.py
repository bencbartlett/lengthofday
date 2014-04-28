# Analysis of a Snowball Earth Model in Breaking Atmospheric Resonance

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Importations
#-----------------------------------------------------------------------------------------------------------------------

from math import *
import numpy as np
import matplotlib.pyplot as plt
import random

#-----------------------------------------------------------------------------------------------------------------------
# Variables
#-----------------------------------------------------------------------------------------------------------------------

# Constants-------------------------------------------------------------------------------------------------------------

# Earth and Atmospheric Properties
g = 9.81                            # m/s^2
yrsec = np.float64(31155690)        # Number of seconds in a year.  float64() used for high precision
h = hnaught = 28644                 # Meters, just played around in wolfram alpha to find a value that works
rho = 1.275                         # Column density of air at STP in kg/m^3
Cp = 1.005                          # Atmospheric specific heat capacity at 300K, in J/(g*K)
T = Tnaught = 287                   # Surface temperature in K
Rearth = 6378100                    # Meters
Fnaught = 4.7 * 10**13              # Heat flux in W, according to http://www.solid-earth.net/1/5/2010/se-1-5-2010.pdf
I = 2*10**37                        # Moment of Inertia of Earth
omeganaught = (g*h)**.5 / Rearth    # Natural resonance frequency of earth's atmosphere, comes out to 2*pi/21hr
tau = 1200000                       # This assumes a Q factor of 100 from David Politzer's email
moonT = -20*Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))
                                    # Note that we use Tnaught here for scaling purposes, not T


# Adjustable Simulation Parameters:-------------------------------------------------------------------------------------

# Sinusoidal Noise:
numSines = 1
gamma = 100000*yrsec                # Frequency modifier for Omega0 = omega0 + deltaOmega * sin(2pi/gamma*t)
deltaOmega = 2*pi/(21*3600*20)      # Amplitude modifier for Omega0 = omega00 + deltaOmega * sin(2pi/gamma*t)

# Snowball Earth Variables
deltaT = 5                         # Temperature change in snowball earth
snowballStart = .5*10**9 * yrsec    # When the snowball earth starts
coolingTime = .3*10**9 * yrsec      # Time it takes to cool down by deltaT
flatTime = .3*10**9 * yrsec         # How long it remains at cooler temperature
warmingTime = 1*10**4 * yrsec      # Time it takes to warm back up
coolingSlope = -deltaT/coolingTime
warmingSlope = deltaT/warmingTime

# Time and Initial Value Parameters
tStep = 1000 * yrsec                # Step size in seconds for time variable
                                    # Note that at the moment, a small step size is required for accurate calculations.
tmax = 2*10**9 * yrsec              # Age of earth in seconds; simulation stops when it reaches this value
omegastart = 2*pi/(5*3600)          # 2pi/5hr - Initial LoD of Earth

# Miscellaneous Parameters
variance = 1.3                      # Error bounds.  To be accepted as stable, 1/variance < omega/(2pi/21) < variance


#-----------------------------------------------------------------------------------------------------------------------
# Subfunctions
#-----------------------------------------------------------------------------------------------------------------------

def resonance(t, deltaOmega, gamma):
    '''Returns current resonance frequency of earth from several functions.'''
    return omeganaught + snowballEarth(t, snowballStart, deltaT)    # + sineNoise(t, deltaOmega, gamma)

def moonTorque(omega):
    '''Returns lunar torque.'''
    return moonT * (omega/(2*pi/(24*3600)))**6                      # This should eventually be replaced by a 1/r^6 term

def atmTorque(omega, t, tau, gamma, deltaOmega):
    '''Returns atmospheric torque following the analytic solution that was solved for.'''
    return -Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - resonance(t, deltaOmega, gamma)**2) + \
        omega/(tau**2)) / (4*(omega**2 - resonance(t, deltaOmega, gamma)**2)**2 + (omega**2)/(tau**2))

def whiteNoise(amplitude, avg):
    '''Returns white noise.'''
    return avg + random.uniform(-amplitude, amplitude)

def resetWave(deltaOmega, gamma):
    '''Prepares the sinusoidal components for the sineNoise() function.'''
    global freq
    global phi
    global amp
    freq = 2*pi*np.random.normal(1, .5, numSines)                   # Random frequency array
    phi = 2*pi*np.random.rand(numSines)                             # Random phase angle array
    amp = deltaOmega/numSines * np.random.normal(0, .5, numSines)   # Random amplitude array

def sineNoise(t, deltaOmega, gamma):
    '''Returns the sum of sine waves to simulate random noise.'''
    return np.sum(amp*np.sin(freq*t/gamma + phi))

def snowballEarth(t, tStart, deltaT):
    '''Returns a resonance frequency  waveform similar to that present in a snowball earth climate model.'''
    global T
    global h
    # Add or subtract temperature to simulate the climate change
    if t >= tStart and t < tStart + coolingTime:
        T += coolingSlope * tStep
    elif t >= tStart + coolingTime and t < tStart + coolingTime + flatTime:
        pass
    elif t >= tStart + coolingTime + flatTime and t < tStart + coolingTime + flatTime + warmingTime:
        T += warmingSlope * tStep
    h = hnaught * T/Tnaught
    omegaTemperatureVariance = omeganaught*(h/hnaught - 1)          # Since h ~ T and omega ~ sqrt(h)
    return omegaTemperatureVariance

def isStable(lastOmega):
    '''Tests to see if the current LoD is near the resonance frequency.  Currently a vestigial function.'''
    if lastOmega/omeganaught < variance and lastOmega/omeganaught > 1/variance:
        return True
    else:
        return False

#-----------------------------------------------------------------------------------------------------------------------
# Main simulation function
#-----------------------------------------------------------------------------------------------------------------------

def simulate(deltaOmega, gamma, tau):
    '''Main simulation loop for 0 < t < tmax.'''
    # Initialize variables
    omega = omegastart
    t = 0
    resetWave(deltaOmega, gamma)
    # Initialize data storage arrays
    omegaTimeValues = []
    omegaValues = []
    dayLengthValues = []
    torqueValues = []
    tempValues = []
    #plotSineNoise()
    while t <= tmax:
        print("Time: %.3f Myr   Omega: %.10f   Day length: %.10f   Temperature: %.10f"\
            % (int(t/(yrsec*1000000)), omega, 2*pi/(3600*omega), T-273))
        domega = (atmTorque(omega, t, tau, gamma, deltaOmega) - moonTorque(omega)) / I * tStep 
        omega += domega                                             # Increment omega
        dayLengthValues.append(2*pi/(3600*omega))                   # Store day length
        omegaValues.append(omega)
        tempValues.append(T-273)
        t += tStep                                                  # Increment 
    omegaF = omegaValues[len(omegaValues)-1]
    plot(dayLengthValues, tempValues)
    return omegaF

def plot(dayValues, tempValues):
    '''Plots the overall LoD and temperature values over time.'''
    fig, ax1 = plt.subplots()
    ax1.plot(dayValues, 'b')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Length of Day (hr)", color='b')
    ax1.set_ylim([0,30])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.plot(tempValues, 'r')
    ax2.set_ylabel('Average Temperature (C)', color='r')
    ax2.set_ylim([Tnaught-273 - 30, Tnaught-273 + 30])              # Set reasonable range and convert K to C
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.show()

def plotSineNoise():
    '''Plots sum of sinusoidal components for visualization purposes.'''
    plt.xlim([0, 2*pi*gamma])
    x = np.linspace(0.0, 10*pi*gamma, 200)
    y = np.zeros(len(x))
    i = 0
    for n in (x):
        y[i] = sineNoise(n, deltaOmega, gamma)
        i+=1
    plt.plot(x, y)
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Main Loop
#-----------------------------------------------------------------------------------------------------------------------

def main():
    '''Main loop over multiple possible variables.'''
    print "Simulating with deltaOmega="+str(deltaOmega)+", gamma="+str(gamma)+"..."
    simulate(deltaOmega, gamma, tau)


#-----------------------------------------------------------------------------------------------------------------------
# Data writing
#-----------------------------------------------------------------------------------------------------------------------

def writedata(omegaValues, dayLengthValues, torqueValues, omegaF, stability, deltaOmega, gamma):
    '''Writes data to a .dat file that should be easily importable into Mathematica.'''

    filename = "Analysis Data  deltaOmega=%.2fw0   gamma=%.0fyr.dat" % (deltaOmega/omeganaught, gamma/yrsec)
    filehandle = open(filename, "w")
    print "Writing data to file for deltaOmega="+str(deltaOmega)+", gamma="+str(gamma)+"..."
    print "Final day length of "+str(2*pi/(3600*omegaF))+" hours.\n"
    filehandle.write("{"+str(deltaOmega)+","+str(gamma)+","+str(omegaF)+","+str(stability)+"}, ")
    filehandle.write(str(dayLengthValues).replace("[","{").replace("]","}")+", ")
    filehandle.write("\n")
    filehandle.close()
    
    filename = "Perturbation Analysis Results.dat"                  # Write to overall cumulative file
    filehandle = open(filename, "a")
    print "Writing data to cumulative file..."
    filehandle.write("{"+str(deltaOmega)+","+str(gamma)+","+str(omegaF)+","+str(stability)+"}, ")
    filehandle.write("\n")
    filehandle.close()


#-----------------------------------------------------------------------------------------------------------------------
# Execute
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
