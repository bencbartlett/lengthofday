# Analysis of a Snowball Earth Model in Breaking Atmospheric Resonance
# Ben Bartlett - Ph11 Research


#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# Importations
#-----------------------------------------------------------------------------------------------------------------------

from math import pi                 #
import numpy as np                  # 
import matplotlib.pyplot as plt     # 
import multiprocessing              # 
import time                         # 

#-----------------------------------------------------------------------------------------------------------------------
# Variables
#-----------------------------------------------------------------------------------------------------------------------

# Constants-------------------------------------------------------------------------------------------------------------

# Earth and Atmospheric Properties
g = 9.81                            # m/s^2
yrsec = np.float64(31155690)        # Number of seconds in a year.  float64() used for high precision
rho = 1.275                         # Column density of air at STP in kg/m^3
Cp = 1.005                          # Atmospheric specific heat capacity at 300K, in J/(g*K)
Rearth = 6378100                    # Meters
Rroche = 9.496 * 10**6              # Solid Roche limit of Earth in m
Vroche = 6.447 * 10**3              # Orbital velocity in m/s at roche limit
Fnaught = 4.7*10**13 / 5.1*10**14   # Heat flux in W/m^2, from http://www.solid-earth.net/1/5/2010/se-1-5-2010.pdf
I = np.float64(8.04*10**37)         # Moment of Inertia of Earth, kgm^2
Mmoon = np.float64(7.347*10**22)    # Lunar mass, kg
hnaught = 28644                     # Meters, just played around in wolfram alpha to find a value that works
omeganaught = (g*hnaught)**.5/Rearth# Natural resonance frequency of earth's atmosphere, comes out to 2*pi/21hr
Tnaught = 287                       # Surface temperature in K

# Lunar torque scaling
moonT = np.float64(6 * 10**16)      # Precambrian lunar torque was 6E23 dyne-cm = 6E16 Nm, according to Zahnle paper
                                    # Scaling factor for atmospheric torque

# Initial total angular momentum of earth moon system
L0 = I*(2*pi/(5*3600)) + Mmoon * Rroche * Vroche



# Adjustable Simulation Parameters:-------------------------------------------------------------------------------------

# Sinusoidal Noise:
numSines = 1                        # Number of sine waves to sum up in atmospheric noise generation
freqVals = 16                       # Number of frequency values to test
ampVals = 16                        # Number of amplitude values
phiVals = 4
# Tvariance = 5                     # Temperature variance in atmospheric variation
gamma = yrsec                       # Frequency modifier for Omega0 = omega0 + deltaOmega * sin(freq*2pi/gamma*t)
deltaOmega = 0 #0*2*pi/(21*3600*20) # Amplitude modifier for Omega0 = omega00 + deltaOmega * sin(2pi/gamma*t)
frequencyEvaluationValues = 1.0 / np.linspace(100, 50000, freqVals)     # Frequency values to test
amplitudeEvaluationValues = np.linspace(0, 50, ampVals)                 # Amplitude values to test
phiEvaluationValues = np.linspace(0, 2*pi, phiVals+1)[:-1] 

# Snowball Earth Variables
deltaT = 25                         # Temperature change in K for snowball earth (singlesimulation only)
snowballStart = 4.54*10**9 * yrsec - (6.5*10**8 + 20*10**6)*yrsec     # When the snowball earth starts
coolingTime = 1 * 1*10**5 * yrsec   # Time it takes to cool down by deltaT
flatTime = 2.5 * 10**7 * yrsec        # How long it remains at cooler temperature
postWarmingTime = 1*10**8 * yrsec   # Padding after global warmup
# warmingTime = 1*10**7 * yrsec     # Time it takes to warm back up
# coolingSlope = -deltaT/coolingTime
# warmingSlope = deltaT/warmingTime

# Time and Initial Value Parameters
tStep = 25000 * yrsec                # Step size in seconds for time variable, small step size for accuracy with high Q
omegastart = 2*pi/(3600*7)         # Starts at resonant frequency
defaultTMax = 2*10**8 * yrsec       # Default value for tmax if snowballTrue = False

# Miscellaneous Parameters
numCores = multiprocessing.cpu_count()  # Gets the number of cores on the computer to optimize number of processes
variance = 1.025                        # Error bounds.  To be stable, 1/variance < omega/(2pi/21*3600) < variance
testSteps = 97                          # Every nth step is tested for monotonic patterns

# Resolution for Q by Tau graph:    # The entire algorithm should run in n^2*log(n) time
Qvals = 30                          # This runs in n time
warmingVals = 30                    # This runs in log(n) time
tempVals = 15                       # This runs in n time
QevaluationValues = np.logspace(np.log10(30), np.log10(500), Qvals)
WevaluationValues = np.logspace(np.log10(1*10**6 * yrsec), np.log10(1*10**9 * yrsec), warmingVals)
TevaluationValues = np.linspace(0, 40, tempVals)  # Temperature should be in increasing order

#-----------------------------------------------------------------------------------------------------------------------
# Subfunctions
#-----------------------------------------------------------------------------------------------------------------------

def resonance(t, deltaOmega, gamma, deltaT, T, h, snowballTrue, snowballStart, coolingTime, coolingSlope, \
    flatTime, warmingTime, warmingSlope, freq, phi, amp):
    '''Returns current resonance frequency of earth from several functions.'''
    if snowballTrue:
        omegaVariance, newT, newh = snowballEarth(t, snowballStart, deltaT, T, h,\
            coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope)
        newT += atmosphericNoise(t, amp, gamma, freq, phi)        # Adds atmospheric temperature variation
        newOmegaNaught = omeganaught + omegaVariance
    else:
        newT = T + atmosphericNoise(t, amp, gamma, freq, phi)
        newh = hnaught * newT/Tnaught
        newOmegaNaught = omeganaught*(newh/hnaught)
    return newOmegaNaught, newT, newh

def moonTorque(omega, t, tmax): # (!) Replace value with correct scaling
    '''Returns lunar torque.'''
    # return -1 * moonT # * (omega/(2*pi/(24*3600)))**6             # This should eventually be replaced by a 1/r^6 term
    # See math, r\propto(L-Iw_earth)^2
    return -1 * moonT * (t/tmax) * np.float64( (L0 - I * (2*pi/(24*3600)) )**2 / ((L0 - I * omega)**2) )**6
    # Moon couldn't have started at roche limit, since it is orbiting faster than earth is rotating, which would speed
    # earth up and slow moon down, causing it to crash into earth
    # The t/tmax is given to take into account the gradually increasing lunar torque over time, likely due to tidal
    # heating of the mantle, outgassing of water from the mantle, formation of oceans, etc.

    # Note that the linear scaling won't really matter for small day length values, since the predominating 1/r^6 term
    # will outweigh this, but accounts for the increase in base lunar torque since the Precambrian period cited in
    # Zahnle and Walker 1987.


def atmTorque(omega, t, tau, deltaT, A24, gamma, deltaOmega, T, h, snowballTrue, snowballStart, coolingTime, \
    coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp):
    '''Returns atmospheric torque following the analytic solution that was solved for.'''
    omega0, newT, newh = resonance(t, deltaOmega, gamma, deltaT, T, h, snowballTrue, snowballStart, coolingTime, \
        coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)
    return moonT/20 * (Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - omega0**2) + omega/(tau**2))\
        / (4*(omega**2 - omega0**2)**2 + (omega**2)/(tau**2))) / A24, newT, newh, omega0

def whiteNoise(amplitude, avg):
    '''Returns white noise.'''
    return avg + np.random.uniform(-amplitude, amplitude)

def resetWave(deltaOmega, gamma):
    '''Prepares the sinusoidal components for the sineNoise() function.'''
    if numSines:
        freq = 2*pi # 2*pi*np.random.normal(1, .5, numSines)            # Random frequency array
        phi = 0 # 2*pi*np.random.rand(numSines)                         # Random phase angle array
        amp = deltaOmega/numSines * np.random.rand(numSines)            # Random amplitude array
        return freq, phi, amp
    else:
        return 0, 0, 0

def sineNoise(t, deltaOmega, gamma, freq, phi, amp):
    '''Returns the sum of sine waves to simulate random noise.'''
    if numSines:
        return np.sum(amp*np.sin(freq*t/gamma + phi))
    else:
        return 0

def atmosphericNoise(t, Tvariance, gamma, freq, phi):
    '''Modifies atmospheric temperature with random noise.'''
    if numSines:
        return Tvariance * np.sum(np.sin(2*pi*freq*t/gamma + phi) - np.sin(2*pi*freq*(t-tStep)/gamma + phi))
    else:
        return 0

def snowballEarth(t, tStart, deltaT, T, h, coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope):
    '''Returns a resonance frequency  waveform similar to that present in a snowball earth climate model.'''
    # Add or subtract temperature to simulate the climate change
    if t >= tStart and t < tStart + coolingTime:
        T += coolingSlope * tStep
    elif t >= tStart + coolingTime and t < tStart + coolingTime + flatTime:
        pass
    elif t >= tStart + coolingTime + flatTime and t < tStart + coolingTime + flatTime + warmingTime:
        T += warmingSlope * tStep
    h = hnaught * T/Tnaught
    omegaTemperatureVariance = omeganaught*(h/hnaught - 1)     # Since h ~ T and omega ~ sqrt(h) ~ h for h/h0 near 1
    return omegaTemperatureVariance, T, h

def hoffmanSnowballEarth(t, tStart, deltaT, T, h, coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope): # (!)
    '''Returns a resonance frequency  waveform similar to that present in a snowball earth climate model.
    Uses snowball Earth model published by Hoffman, et al.'''
    # Add or subtract temperature to simulate the climate change
    if t >= tStart and t < tStart + coolingTime:
        T += coolingSlope * tStep
    elif t >= tStart + coolingTime and t < tStart + coolingTime + flatTime:
        pass
    elif t >= tStart + coolingTime + flatTime and t < tStart + coolingTime + flatTime + warmingTime:
        T += warmingSlope * tStep
    h = hnaught * T/Tnaught
    omegaTemperatureVariance = omeganaught*(h/hnaught - 1)     # Since h ~ T and omega ~ sqrt(h) ~ h for h/h0 near 1
    return omegaTemperatureVariance, T, h

def isStable(lastOmega):
    '''Tests to see if the current LoD is near the resonance frequency.'''
    if omeganaught/lastOmega < variance and omeganaught/lastOmega > 1/variance: # Tests to see if it's near resonance
        return True
    else:
        return False

# def isStableNoiseInclusive(lastOmega, omega0): # (!) Fix this to solve for stability correctly every time
#     '''Tests to see if the final LoD is near the resonance frequency while including the changes in resonance freq
#     induced by atmospheric fluctuations.'''
#     if omega0/lastOmega < variance and omega0/lastOmega > 1/variance:
#         return True
#     else:
#         return False

# def isStableNoiseInclusive(lastOmega, minOmega0): # (!) Fix this to solve for stability correctly every time
#     '''Revised stability algorithm setting an upper bound on the LoD to examine whether it is past that point.
#     Not currently functional for some reason.'''
#     lastLoD = 2*pi/(3600*lastOmega)
#     maxLoD = 2*pi/(3600*minOmega0)
#     if lastLoD > maxLoD * 1.01:
#         return True
#     else:
#         return False

def isStableNoiseInclusive(omegaValues, searchLength, printTrue = False):
    '''Revised revised stability algorithm examining whether the last x omega values are strictly decreasing,
    where x is some multiple of the noise period greater than 1.  After fairly extensive testing, this also seems
    to very accurately measure the stability of the end result.'''
    if printTrue:
        print "Testing omegaValues for monotonic ordering..."
    for i in range(1, searchLength+1):
        if printTrue:
            print str(np.around(omegaValues[-testSteps * (i+1)], 10))+" <- " + \
                str(np.around(omegaValues[-testSteps * i], 10))
        if np.around(omegaValues[-testSteps * (i+1)], 10) <= np.around(omegaValues[-testSteps * i], 10):  
            # Checks backwards for strict decreasingness
            return True  # If earth has sped up or remained the same, it's in resonance
    return False

#-----------------------------------------------------------------------------------------------------------------------
# Auxilliary Functions
#-----------------------------------------------------------------------------------------------------------------------

def invisibleSpines(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

def plotData(dayValues, tempValues, tmax, title, atmTorques = False, lunarTorques = False):
    '''Plots the overall LoD and temperature values over time.'''
    timescale = np.linspace(0, (tmax/yrsec)/1000000, len(dayValues))
    lowerDayLength = 5
    upperDayLength = 25
    fig, ax1 = plt.subplots()
    # Plot temperature values
    ax2 = ax1.twinx()
    ax2.plot(timescale, tempValues, 'r')
    ax2.set_ylabel('Average Temperature (C)', color='r')
    ax2.set_ylim([Tnaught-273 - 100, Tnaught-273 + 50])              # Set reasonable range and convert K to C
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    # Plot day length values
    ax1.plot(timescale, dayValues, 'b')
    ax1.set_xlabel("Time (Myr)")                             
    ax1.set_ylabel("Length of Day (hr)", color='b')
    ax1.set_ylim([lowerDayLength, upperDayLength])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    if atmTorques:
        ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("axes", 1.2))
        # invisibleSpines(ax3)
        # ax3.spines["right"].set_visible(True)
        ax3.plot(timescale, atmTorques, 'm')
        ax3.set_ylabel('Atmospheric Torques (Nm)', color='m')
        ax3.set_ylim([2 * np.min(atmTorques), 3 * 2 * np.max(atmTorques)]) 
        for tl in ax3.get_yticklabels():
            tl.set_color('m')

    if lunarTorques:
        ax4 = ax1.twinx()
        ax4.plot(timescale, lunarTorques, 'g')
        ax4.set_ylabel('Lunar Torques (Nm)', color='g')
        ax4.set_ylim([0, np.min(lunarTorques)]) 
        for tl in ax4.get_yticklabels():
            tl.set_color('g')

    # Show plot
    plt.xlim([0, 4500])
    plt.title(title)
    plt.show()

def plotSineNoise(Tvariance, gamma, freq, phi):
    '''Plots sum of sinusoidal components for visualization purposes.'''
    lbound = 0
    ubound = 1000*2*pi*gamma
    plt.xlim([lbound, ubound])
    x = np.linspace(lbound, ubound, 200)
    y = np.zeros(len(x))
    i = 0
    for n in (x):
        y[i] = atmosphericNoise(n, Tvariance, gamma, freq, phi)     # Gets atmospheric torque for visualization
        i+=1
    plt.plot(x, y)
    plt.show()

def plotTorque(Q):
    '''Test function; plots atmospheric and lunar torque as a function of omega'''
    t = 0                                                           # Reset time
    T = Tnaught                                                     # Reset temperature
    h = hnaught                                                     # Reset atmospheric height
    freq, phi, amp = resetWave(deltaOmega, gamma)                   # Reset noise modifiers
    # Do some calculations
    tau = Q / omeganaught
    coolingSlope = 0
    warmingSlope = 0
    warmingTime = 10**99
    # Current atmospheric column height
    A24 = Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))

    moonTorques = []
    atmTorques = []

    lowerTimeLim = 5                                                # Lower LoD time limit, hr.
    upperTimeLim = 24                                               # Upper LoD time limit, hr.

    for dayLength in np.linspace(lowerTimeLim, upperTimeLim, 1000):
        omega = (2*pi)/(3600*dayLength)
        moonTorques.append(-moonTorque(omega, 1, 1))
        atmTorques.append(atmTorque(omega, t, tau, deltaT, A24, gamma, deltaOmega, T, h, False, snowballStart, \
            coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)[0])

    plt.xlim([lowerTimeLim, upperTimeLim])
    plt.plot(np.linspace(lowerTimeLim, upperTimeLim, 1000), atmTorques, label = "Atmospheric torque")
    plt.plot(np.linspace(lowerTimeLim, upperTimeLim, 1000), moonTorques, label = "-1 * Lunar torque")
    plt.xlabel("Length of Day (hr)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.show()

def sortClean(array, xdim, ydim):
    '''Sorts the processing results to make sure they are in order and cleans up the data by removing indices.'''
    newarray = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            newarray[array[i][j][1]][array[i][j][2]] = array[i][j][0]    # Sort the arrays by indices
    return newarray

#-----------------------------------------------------------------------------------------------------------------------
# Data writing
#-----------------------------------------------------------------------------------------------------------------------

def writedata(omegaValues, dayLengthValues, torqueValues, omegaF, stability, deltaOmega, gamma):
    '''Writes data to a .dat file that should be easily importable into Mathematica.
    Somewhat outdated at the moment.'''

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

def stabilityBoundary(stabilityValues):
    '''Exports a 3D heightmap file for stability-preserving Q-deltaT-tw tuples for export into Mathematica.'''
    stabilityBoundary = np.zeros((Qvals, warmingVals))
    for T in range(tempVals):
        for Q in range(Qvals):
            # This sets the (Q,tw) element of the projection array to the current T value, finding the boundary,
            # allowing it to be used with functions such as the ListPlot3D[] function in Mathematica.
            stabilityBoundary[Q, np.nonzero(stabilityValues[T,Q])[0]] = TevaluationValues[T]
    return stabilityBoundary

def writeStabilityData(stabilityValues, Qaxis, Waxis):
    '''Writes individual "slices" of the stabilityBoundary file as raw data, since the conversion is possibly lossy.'''
    np.savetxt('StabilityRegime.dat', stabilityBoundary(stabilityValues), fmt = '%i', delimiter = ',') # Write boundary
    for T in range(tempVals):
        np.savetxt('StabilityRegimeT=%d.dat' % int(TevaluationValues[T]), stabilityValues[T], \
            fmt = '%i', delimiter = ',')                            # Exports individual array slices to CSV format
    filehandle = open("AxesLabels.txt", "w")
    filehandle.write("FrameTicks->{"+Qaxis+","+Waxis+"}")
    filehandle.close()

def writeNoiseStabilityData(stabilityValues, Faxis, Aaxis, Paxis):
    '''Separate writeStabilityData function for sinusoidally-driven atmospheric temperature simulations.'''
    for phi in range(phiVals):
        np.savetxt('NoiseStabilityRegimePhi=%.3f.dat' % phiEvaluationValues[phi], stabilityValues[phi], \
            fmt = '%i', delimiter = ',')
    filehandle = open("NoiseAxesLabels.txt", "w")
    filehandle.write("FrameTicks->{"+Paxis+","+Faxis+","+Aaxis+"}")
    filehandle.close()

def showStabilityData(stabilityValues, Qaxis, Waxis):
    '''Displays a pretty plot. :P'''
    Qstr = ["%.1e" % val for val in Qaxis]
    Wstr = ["%.1e" % val for val in Waxis]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data)

    ax.set_xticklabels(['']+Wstr)
    ax.set_xlabel('Tau_W')
    ax.set_yticklabels(['']+Qstr)
    ax.set_ylabel('Q')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Main simulation functions
#-----------------------------------------------------------------------------------------------------------------------

def simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, \
    noiseFrequency, noiseAmplitude, noisePhi, queue, i, j, k, printTrue, plotTrue, putToQueue, snowballTrue, isNoise):
    '''Main simulation loop for 0 < t < tmax.'''
    # Reset variables
    omega = omegastart                                              # Reset angular velocity
    t = 0                                                           # Reset time
    T = Tnaught                                                     # Reset temperature
    h = hnaught                                                     # Reset atmospheric height
    #freq, phi, amp = resetWave(deltaOmega, gamma)                  # Reset noise modifiers
    freq = noiseFrequency
    amp = noiseAmplitude
    phi = noisePhi

    # Do some calculations
    if snowballTrue:
        tmax = snowballStart + coolingTime + flatTime + warmingTime + postWarmingTime  # Total simulation time
    else:
        tmax = defaultTMax

    tau = Q / omeganaught
    coolingSlope = np.float64(-deltaT/coolingTime)
    warmingSlope = np.float64(deltaT/warmingTime)
    minOmega0 = float("inf")
    # Initialize data storage arrays
    omegaTimeValues = []
    omegaValues = []
    dayLengthValues = []
    torqueValues = []
    tempValues = []

    # Current atmospheric column height
    A24 = Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))
    
    counter = 0
    while t <= tmax:
        if counter == 1000 and printTrue:
            print("Time: %.3f Myr   Omega: %.10f   Day length: %.10f   Temperature: %.10f"\
                % ((t/(yrsec*1000000)), omega, 2*pi/(3600*omega), T-273))
            counter = 0


        atmTorqueResults, newT, newh, omega0 = atmTorque(omega, t, tau, deltaT, A24, gamma, deltaOmega, T, h, \
            snowballTrue, snowballStart, coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)

        if omega0 < minOmega0:
            minOmega0 = omega0

        domega = (atmTorqueResults + moonTorque(omega, t, tmax)) / I * tStep 
        omega += domega                                                 # Increment omega
        T, h = newT, newh                                               # Update temperature and column height values

        dayLengthValues.append(2*pi/(3600*omega))                       # Store day length
        omegaValues.append(omega)
        tempValues.append(T - 273)

        t += tStep                                                      # Increment 
        counter += 1


    if printTrue:
        print "Stability: " + str(isStableNoiseInclusive(omegaValues, 2*stepPeriod, True))

    if plotTrue:
        plotData(dayLengthValues, tempValues, tmax, "Index: ("+str(i)+","+str(j)+\
            "), Period: "+str(int(1/freq))+", Amp: "+str(amp))

    if putToQueue:
        if isNoise:
            queue.put([isStableNoiseInclusive(omegaValues, 2*stepPeriod), i, j, k])
        else:
            queue.put([isStable(omegaValues[-1]), i, j, k])             # For multiprocessing, return value to queue

    if isNoise:
        return isStableNoiseInclusive(omegaValues, 2*stepPeriod)        # Temperature inclusive values for noise regimes
    else:
        return isStable(omegaValues[-1])                                # Get last omega value of simulation

#-----------------------------------------------------------------------------------------------------------------------
# LoD Over Earth's History - Used in generating figure in paper
#-----------------------------------------------------------------------------------------------------------------------

def lodHistory(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, \
    noiseFrequency, noiseAmplitude, noisePhi, queue, i, j, k, printTrue, plotTrue, putToQueue, snowballTrue, isNoise,
    dataDump):
    '''Main simulation loop for 0 < t < tmax.'''
    # Reset variables
    omega = omegastart                                              # Reset angular velocity
    t = 0                                                           # Reset time
    T = Tnaught                                                     # Reset temperature
    h = hnaught                                                     # Reset atmospheric height
    #freq, phi, amp = resetWave(deltaOmega, gamma)                  # Reset noise modifiers
    freq = noiseFrequency
    amp = noiseAmplitude
    phi = noisePhi

    stepPeriod = int(10000000 * yrsec / (tStep * testSteps))        # Checks the last 10M years for stability

    # Do some calculations
    tmax = 4.52 * 10**9 * yrsec                                     # Age of Moon in years
    tau = Q / omeganaught
    coolingSlope = np.float64(-deltaT/coolingTime)
    warmingSlope = np.float64(deltaT/warmingTime)
    minOmega0 = float("inf")
    # Initialize data storage arrays
    omegaTimeValues = []
    omegaValues = []
    dayLengthValues = []
    atmTorqueValues = []
    lunarTorqueValues = []
    tempValues = []

    # Current atmospheric column height
    A24 = Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))
    
    counter = 0
    while t <= tmax:
        if counter == 100 and printTrue:
            print("Time: %.3f Myr   Omega: %.10f   Day length: %.10f   Temperature: %.10f"\
                % ((t/(yrsec*1000000)), omega, 2*pi/(3600*omega), T-273))
            counter = 0

        atmTorqueResults, newT, newh, omega0 = atmTorque(omega, t, tau, deltaT, A24, gamma, deltaOmega, T, h, \
            snowballTrue, snowballStart, coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)

        if omega0 < minOmega0:
            minOmega0 = omega0

        domega = (atmTorqueResults + moonTorque(omega, t, tmax)) / I * tStep 

        atmTorqueValues.append(atmTorqueResults)
        lunarTorqueValues.append(moonTorque(omega, t, tmax))

        omega += domega                                                 # Increment omega
        T, h = newT, newh                                               # Update temperature and column height values

        dayLengthValues.append(2*pi/(3600*omega))                       # Store day length
        omegaValues.append(omega)
        tempValues.append(T - 273)

        t += tStep                                                      # Increment 
        counter += 1

    # Plot the day length
    plotData(dayLengthValues, tempValues, tmax, "Length of Day over Earth's History", \
        atmTorques = atmTorqueValues, lunarTorques = lunarTorqueValues)

    if dataDump:
        np.savetxt('DayLengthValues.dat', dayLengthValues[::1000], delimiter = ',', newline = ",")
        np.savetxt('TempValues.dat', tempValues[::1000], delimiter = ',', newline = ",")
        np.savetxt('AtmTorqueValues.dat', atmTorqueValues[::1000], delimiter = ',', newline = ",")
        np.savetxt('LunarTorqueValues.dat', lunarTorqueValues[::1000], delimiter = ',', newline = ",")



#-----------------------------------------------------------------------------------------------------------------------
# Binary Search for Stability-Instability Boundary
#-----------------------------------------------------------------------------------------------------------------------

def binarySearchRegime(Q, deltaT, Qindex, Tindex, queue, printTrue = False, putToQueue = False):
    '''Same thing as the regimeSimulation() function, except instead of simulating every value; it searches for
    the boundary of the stability-nonstability region by iterating over warmingTime for a given Q and deltaT.'''
    lowerBound = 0
    upperBound = warmingVals - 1
    wPosition = int(np.mean([lowerBound, upperBound]))                  # Start at the middle of the w-values array
    while True:
        # Simulate the function to see if it's stable or not, iterate over the binary search
        if printTrue:
            print("Starting simulation thread for Q = %.3e, Tau_w = %.3es = %.3e yr." % \
                (Q, WevaluationValues[wPosition], WevaluationValues[wPosition]/yrsec))

        result = simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, \
            WevaluationValues[wPosition], 2*pi*0, 0, 0, queue, Qindex, Tindex, 0, \
            printTrue = False, plotTrue = False, putToQueue = False, snowballTrue = True, isNoise = False)

        if result == True:                      # If simulation is stable, everything right of it ("above" it) is stable
            upperBound = wPosition
            if wPosition == int(np.mean([lowerBound, upperBound])):     # If no change in position
                if putToQueue:
                    queue.put([Tindex, Qindex, wPosition])
                return wPosition   # Returns the last stable value
            wPosition = int(np.mean([lowerBound, upperBound]))

        else:                                   # If simulation is not stable, everything "below" it is not stable
            lowerBound = wPosition
            if wPosition == int(np.mean([lowerBound, upperBound])):     # If no change in position
                if putToQueue:
                    queue.put([Tindex, Qindex, wPosition])
                return wPosition  # Returns the last stable value
            wPosition = int(np.mean([lowerBound, upperBound]))

#-----------------------------------------------------------------------------------------------------------------------
# Q-deltaT-tw Stability Regime Simulation
#-----------------------------------------------------------------------------------------------------------------------

def regimeSimulation():
    '''Simulates through a large number of variable combinations to find an area of stability-preserving conditions.'''
    queue = multiprocessing.Queue() # Initialize a multiprocessing queue for communication between processes
    startTime = time.clock()        # Timing purposes, though the prediction algoriths don't work at the moment

    stabilityArray = np.zeros((tempVals, Qvals, warmingVals))       # Initial unsorted array output
    i = j = k = currentProcesses = 0                                # Counter variables to track position
    iPos = jPos = 0                                                 # Position variables to track array placement

    for deltaT in TevaluationValues:                                # Uses k index
        stabilityQaxis = "{{"                                       # Mathematica axis strings
        i = iPos = 0
        for Q in QevaluationValues:                                 # Uses i index
            # Start the binary search fo a given Q, T pair
            print("Starting binary search for Q = %.3e, deltaT = %.3eK." % (Q, deltaT))
            # Start a simulation process
            process = multiprocessing.Process(target = binarySearchRegime, args=(Q, deltaT, i, k, queue, True, True,))
            process.start()                                         # Start the process 
            currentProcesses += 1                                   # Increment number of processes

            # Retrieve results
            if currentProcesses == numCores or (i==Qvals-1 and k==tempVals-1): # Gets the values if full or finished
                recoveryAttempts = currentProcesses                 # Set the no. of elements to recover to the proc.no.
                print "\n"
                for index in np.arange(recoveryAttempts):
                    print("  > Attempting to recover array row %d..." % iPos)
                    result = queue.get()                            # Get the unsorted array of unordered results
                    if result[2] < (warmingVals-1) - 1:             # Accounts for computational hangups of nonresonance
                        stabilityArray[result[0], result[1], result[2]:] = 1 # Sets all w values above bound to stable
                    print ("    >> Array element recovered for (Q,dT) = (%d,%d). Set all tw values above %d to stable."\
                        % (result[1], result[0], result[2]) )
                    iPos += 1
                currentProcesses = 0                                # Reset number of processes
                print "\n"

            stabilityWaxis = "{{"                                   # For mathematica label generation

            for warmingTime in WevaluationValues:                   # Uses j index
                if j % 8 == 0:
                    stabilityWaxis += ("{%d, %.1e}," % (j+1, warmingTime/yrsec))

            if i % 4 == 0:
                stabilityQaxis += ("{%d, %.1e}," % (i+1, Q))
            j = 0                                                   # Reset index variable
            i += 1                                                  # Increment

        k += 1
        timeEstimate = (time.clock() - startTime)/(60) * (tempVals/(k+1) - 1)
        timeHr = timeEstimate / 60
        timeMin = timeEstimate - 60 * timeHr
        print("---> Simulation %.0d%% complete. Estimated time remaining: %i hr, %i min.\n" \
                % (int(100.0 * k / tempVals),  timeHr, timeMin) )   # Give progress report

    stabilityQaxis = stabilityQaxis[:-1] + "}, None}"               # Label for Q axis
    stabilityWaxis = stabilityWaxis[:-1] + "}, None}"               # Label for W axis

    # print "Parsing array..."
    # stabilityArray = sortClean(stabilityArray, Qvals, warmingVals) # Sort the array to make sure it is in order

    writeStabilityData(stabilityArray, stabilityQaxis, stabilityWaxis)
    # showStabilityData(stabilityArray, QevaluationValues, WevaluationValues)
    raw_input("Simulation complete. %d of the %d simulations preserved resonance.\nPress enter to close this window." \
        % (np.count_nonzero(stabilityArray), Qvals*warmingVals*tempVals))

#-----------------------------------------------------------------------------------------------------------------------
# Atmospheric Noise Stability Regime
#-----------------------------------------------------------------------------------------------------------------------

def noiseRegimeSimulation(): 
    '''This function iterates over a single plausible but definitely stability-preserving (Q, tw, deltaT) instance
    to analyze how the amplitude and frequency of a sinusoidal driving frequency and/or pink noise can affect the
    stability of the system.  Outputs a 2D array of stability values along the domain of amplitude and frequency.'''
    
    queue = multiprocessing.Queue()             # Initialize a multiprocessing queue for communication between processes
    startTime = time.clock()

    stabilityArray = np.zeros((phiVals, freqVals, ampVals))         # Initial unsorted array output
    i = j = k = currentProcesses = 0                                # Counter variables to track position
    iPos = jPos = kPos = 0                                          # Position variables to track array placement

    Q = 100                                                         # Q factor of the atmosphere
    warmingTime = 2 * 10**7 * yrsec

    showData = False

    for noisePhi in phiEvaluationValues:                            # Uses k index
        stabilityPaxis = "{{"
        for noiseFrequency in frequencyEvaluationValues:            # Uses i index
            stabilityFaxis = "{{"                                   # More Mathematica strings
            for noiseAmplitude in amplitudeEvaluationValues:        # Uses j index

                print("Starting simulation thread for oscillation period = %d, amplitude = %d C" \
                    % (int(1/noiseFrequency), int(noiseAmplitude)))
                # Start a simulation process
                process = multiprocessing.Process(target = simulate, \
                    args = (deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, \
                    noiseFrequency, noiseAmplitude, noisePhi, queue, i, j, k, False, showData, True, False, True,))
                process.start()                                     # Start the process    
                currentProcesses += 1                               # Increment number of processes

                if currentProcesses == numCores or (i==freqVals-1 and j==ampVals-1): # Gets the values if finished
                    recoveryAttempts = currentProcesses
                    for index in np.arange(recoveryAttempts):
                        print("  > Attempting to recover array element (%d,%d)..." % (iPos, jPos))
                        result = queue.get()                        # Get the unsorted array of unordered results
                        stabilityArray[result[3]][result[1]][result[2]] = result[0]
                        print "  >> Array element recovered.  Result is (%d, %d, %d,) = %d." % \
                            (result[3], result[1], result[2], result[0])
                        jPos += 1
                        if jPos > ampVals - 1:
                            jPos = 0
                            iPos += 1
                    currentProcesses = 0                            # Reset number of processes

                j += 1                                              # Increment j.  The placement of this matters.

                stabilityAaxis = "{{"
                for noiseAmplitude in amplitudeEvaluationValues:    # Uses j index
                    if j % 8 == 0:                                  # j%8 is used because labels become obtrusive
                        stabilityAaxis += ("{%d, %.1e}," % (j+1, noiseAmplitude))

            if i % 4 == 0:
                stabilityFaxis += ("{%d, %.1e}," % (i+1, noiseFrequency))

            j = 0                                                   # Reset index variable
            i += 1                                                  # Increment

            print("---> Simulation %.0d%% complete. \n" \
                % (int(100.0 * i/freqVals * (k+1) / phiVals)) )     # Give progress report

        i = 0
        k += 1
        if k % 4 == 0:
            stabilityPaxis += ("{%d, %.1e}," % (i+1, noisePhi))

    stabilityFaxis = stabilityFaxis[:-1] + "}, None}"
    stabilityAaxis = stabilityAaxis[:-1] + "}, None}"
    stabilityPaxis = stabilityPaxis[:-1] + "}, None}"

    # print "Parsing array..."
    # stabilityArray = sortClean(stabilityArray, Qvals, warmingVals) # Sort the array to make sure it is in order

    writeNoiseStabilityData(stabilityArray, stabilityFaxis, stabilityAaxis, stabilityPaxis)
    # showStabilityData(stabilityArray, QevaluationValues, WevaluationValues)
    raw_input("Simulation complete. %d of the %d simulations preserved resonance.\nPress enter to close this window." \
        % (np.count_nonzero(stabilityArray), freqVals*ampVals*phiVals))

#-----------------------------------------------------------------------------------------------------------------------
# Single simulations for testing purposes
#-----------------------------------------------------------------------------------------------------------------------

def singleSimulation():
    '''Mainly for testing purposes - runs a single instance of the simulate() function.'''
    # print "Simulating with deltaOmega=" + str(deltaOmega) + ", gamma=" + str(gamma) + "..."
    queue = multiprocessing.Queue()
    Q = 100  # Q factor of the atmosphere
    warmingTime = 5 * 10**7 * yrsec
    i=j=k=0
    plotTorque(Q)
    simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, \
     1.0/5000000, 0, 0, queue, i, j, k, \
     printTrue = True, plotTrue = True, putToQueue = False, snowballTrue = True, isNoise = False)

def historySimulation():
    '''Invokes the LoD/history generation function.'''
    # print "Simulating with deltaOmega=" + str(deltaOmega) + ", gamma=" + str(gamma) + "..."
    queue = multiprocessing.Queue()
    Q = 100  # Q factor of the atmosphere
    warmingTime = 5 * 10**7 * yrsec
    i=j=k=0
    frequency = np.array([1.0/50000000, 1.0/6000000, 1.0/7000000, 1.0/80000000, 1.0/130000000, 1.0/425400000, \
        1.0/730000000])
    lodHistory(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, \
     frequency, .5, 0, queue, i, j, k, \
     printTrue = True, plotTrue = True, putToQueue = False, snowballTrue = True, isNoise = False, dataDump = True)

def singleBinarySearch():
    '''Single binary search implementation for testing purposes.'''
    deltaT = 25
    Q = 100
    queue = multiprocessing.Queue()
    binarySearchRegime(Q, deltaT, 0, 0, queue, True, False)

#-----------------------------------------------------------------------------------------------------------------------
# Execute
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #singleSimulation()
    historySimulation()
    #singleBinarySearch()
    #regimeSimulation()
    #noiseRegimeSimulation()
