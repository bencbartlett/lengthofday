# Analysis of a Snowball Earth Model in Breaking Atmospheric Resonance

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Importations
#-----------------------------------------------------------------------------------------------------------------------

from math import pi                # Cause I get tired of typing np.pi over and over and over agin...
import numpy as np                 # We kind of need this
import matplotlib.pyplot as plt    # Oooh, pretty graphs!
import multiprocessing             # For super-duper-mega-speedarific(TM) computations


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
Fnaught = 4.7 * 10**13              # Heat flux in W, according to http://www.solid-earth.net/1/5/2010/se-1-5-2010.pdf
I = 8.0*10**37                      # Moment of Inertia of Earth
hnaught = 28644                     # Meters, just played around in wolfram alpha to find a value that works
omeganaught = (g*hnaught)**.5/Rearth# Natural resonance frequency of earth's atmosphere, comes out to 2*pi/21hr
Q = 100                             # Q factor of the atmosphere
tau = Q / omeganaught               # This assumes a Q factor of 100 from David Politzer's email
Tnaught = 287                       # Surface temperature in K

# Lunar torque scaling
moonT = -20*Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))
                                    # Lunar torque is currently (negative) 20 times the atmospheric torque
                                    # Note that we use Tnaught here for scaling purposes, not T


# Adjustable Simulation Parameters:-------------------------------------------------------------------------------------

# Sinusoidal Noise:
numSines = 0
gamma = 1*10**7 * yrsec             # Frequency modifier for Omega0 = omega0 + deltaOmega * sin(2pi/gamma*t)
deltaOmega = 0*2*pi/(21*3600*20)    # Amplitude modifier for Omega0 = omega00 + deltaOmega * sin(2pi/gamma*t)


# Snowball Earth Variables
deltaT = 25                         # Temperature change in snowball earth
snowballStart = .1*10**9 * yrsec    # When the snowball earth starts
coolingTime = .2*10**9 * yrsec      # Time it takes to cool down by deltaT
flatTime = .1*10**9 * yrsec         # How long it remains at cooler temperature
# warmingTime = 1*10**7 * yrsec       # Time it takes to warm back up
# coolingSlope = -deltaT/coolingTime
# warmingSlope = deltaT/warmingTime


# Time and Initial Value Parameters
tStep = 500 * yrsec                 # Step size in seconds for time variable
                                    # Note that at the moment, a small step size is required for accurate calculations.
tmax = 0.6*10**9 * yrsec            # Age of earth in seconds; simulation stops when it reaches this value
omegastart = 2*pi/(21.06*3600)      # 2pi/5hr - Initial LoD of Earth


# Miscellaneous Parameters
variance = 1.05                     # Error bounds.  To be accepted as stable, 1/variance < omega/(2pi/21) < variance



#-----------------------------------------------------------------------------------------------------------------------
# Subfunctions
#-----------------------------------------------------------------------------------------------------------------------

def resonance(t, deltaOmega, gamma, T, h, coolingTime, coolingSlope, \
    flatTime, warmingTime, warmingSlope, freq, phi, amp):
    '''Returns current resonance frequency of earth from several functions.'''
    snowballResults, newT, newh = snowballEarth(t, snowballStart, T, h,\
        coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope)
    return omeganaught + snowballResults + sineNoise(t, deltaOmega, gamma, freq, phi, amp), newT, newh

def moonTorque(omega):
    '''Returns lunar torque.'''
    return moonT * (omega/(2*pi/(24*3600)))**6                      # This should eventually be replaced by a 1/r^6 term

def atmTorque(omega, t, tau, gamma, deltaOmega, T, h, coolingTime, \
    coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp):
    '''Returns atmospheric torque following the analytic solution that was solved for.'''
    omega0, newT, newh = resonance(t, deltaOmega, gamma, T, h, coolingTime, coolingSlope, \
        flatTime, warmingTime, warmingSlope, freq, phi, amp)
    return -Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - omega0**2) + omega/(tau**2))\
        / (4*(omega**2 - omega0**2)**2 + (omega**2)/(tau**2)), newT, newh

def whiteNoise(amplitude, avg):
    '''Returns white noise.'''
    return avg + np.random.uniform(-amplitude, amplitude)

def resetWave(deltaOmega, gamma):
    '''Prepares the sinusoidal components for the sineNoise() function.'''
    if numSines:
        freq = 2*pi*np.random.normal(1, .5, numSines)               # Random frequency array
        phi = 2*pi*np.random.rand(numSines)                         # Random phase angle array
        amp = deltaOmega/numSines * np.random.rand(numSines)        # Random amplitude array
        return freq, phi, amp
    else:
        return 0, 0, 0

def sineNoise(t, deltaOmega, gamma, freq, phi, amp):
    '''Returns the sum of sine waves to simulate random noise.'''
    if numSines:
        return np.sum(amp*np.sin(freq*t/gamma + phi))
    else:
        return 0

def snowballEarth(t, tStart, T, h, coolingTime, coolingSlope, flatTime, warmingTime, warmingSlope):
    '''Returns a resonance frequency  waveform similar to that present in a snowball earth climate model.'''
    # Add or subtract temperature to simulate the climate change
    if t >= tStart and t < tStart + coolingTime:
        T += coolingSlope * tStep
    elif t >= tStart + coolingTime and t < tStart + coolingTime + flatTime:
        pass
    elif t >= tStart + coolingTime + flatTime and t < tStart + coolingTime + flatTime + warmingTime:
        T += warmingSlope * tStep
    h = hnaught * T/Tnaught
    omegaTemperatureVariance = omeganaught*(h/hnaught - 1)          # Since h ~ T and omega ~ sqrt(h)
    return omegaTemperatureVariance, T, h

def isStable(lastOmega):
    '''Tests to see if the current LoD is near the resonance frequency.'''
    if omeganaught/lastOmega < variance and omeganaught/lastOmega > 1/variance: # Tests to see if it's near resonance
        return True
    else:
        return False


#-----------------------------------------------------------------------------------------------------------------------
# Auxilliary Functions
#-----------------------------------------------------------------------------------------------------------------------

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

def sortClean(array, xdim, ydim):
    '''Sorts the processing results to make sure they are in order and cleans up the data by removing indices.'''
    newarray = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            newarray[array[i][j][1]][array[i][j][2]] = array[i][j][0] # Sort the arrays by indices
    return newarray

    

#-----------------------------------------------------------------------------------------------------------------------
# Main simulation function
#-----------------------------------------------------------------------------------------------------------------------

def simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, queue, i, j, plotTrue):
    '''Main simulation loop for 0 < t < tmax.'''
    # Reset variables
    omega = omegastart                                              # Reset angular velocity
    t = 0                                                           # Reset time
    T = Tnaught                                                     # Reset temperature
    h = hnaught                                                     # Reset atmospheric height
    freq, phi, amp = resetWave(deltaOmega, gamma)                   # Reset noise modifiers
    # moonTorqueScalar(tau)                                         # Reset lunar torque modifier\

    # Do some calculations
    tau = Q / omeganaught
    coolingSlope = np.float64(-deltaT/coolingTime)
    warmingSlope = np.float64(deltaT/warmingTime)
    # Initialize data storage arrays
    omegaTimeValues = []
    omegaValues = []
    dayLengthValues = []
    torqueValues = []
    tempValues = []
    # plotSineNoise()

    # print tau, deltaT, snowballStart, coolingTime, flatTime, warmingTime, queue, i, j, plotTrue
    
    counter = 0
    while t <= tmax:
        # if counter == 10000:
        #     print("Time: %.3f Myr   Omega: %.10f   Day length: %.10f   Temperature: %.10f"\
        #         % ((t/(yrsec*1000000)), omega, 2*pi/(3600*omega), T-273))
        #     counter = 0

        atmTorqueReults, newT, newh = atmTorque(omega, t, tau, gamma, deltaOmega, T, h, coolingTime, \
            coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)
        domega = (atmTorqueReults - moonTorque(omega)) / I * tStep 
        omega += domega                                             # Increment omega
        T, h = newT, newh                                           # Update temperature and column height values

        dayLengthValues.append(2*pi/(3600*omega))                   # Store day length
        omegaValues.append(omega)
        tempValues.append(T - 273)

        t += tStep                                                  # Increment 
        counter += 1
    
    if plotTrue:
        plot(dayLengthValues, tempValues)

    queue.put([isStable(omegaValues[-1]), i, j])                    # For multiprocessing, put the return value to queue                   
    return isStable(omegaValues[-1])                                # Get last omega value of simulation


#-----------------------------------------------------------------------------------------------------------------------
# Main Loop
#-----------------------------------------------------------------------------------------------------------------------

def regimeSimulation():
    '''Simulates through a large number of variable combinations to find an area of stability-preserving conditions.'''
    queue = multiprocessing.Queue() # Initialize a multiprocessing queue for communication between processes
    numCores = multiprocessing.cpu_count() # Gets the number of cores on the computer to optimize number of processes

    Qvals = 96 # For the moment, these need to be an integer multiple of numCores
    warmingVals = 96
    stabilityArray = np.zeros((Qvals, warmingVals), dtype = (int, 3)) # Initial unsorted array output
    i = j = 0 # Counter variables to track position

    stabilityQaxis = "{{" # These strings track the axes for easy input into Mathematica

    for Q in np.logspace(np.log10(10000), np.log10(30), Qvals):
        stabilityWaxis = "{{"
        for warmingTime in np.logspace(np.log10(1000 * yrsec), np.log10(1*10**8 * yrsec), warmingVals):

            stabilityWaxis += ("{%d, %.1e}," % (j+1, warmingTime/yrsec))

            # print Q, warmingTime
            # if j % numCores == numCores - 1: # Gets the values if full or finished
            #     for index in np.arange(numCores):
            #         print("  > Attempting to recover array element %d, %d..." % (i, j+index-numCores))
            #         stabilityArray[i][j+index-numCores] = queue.get() # Get the unsorted array of possibly out of order results
            #         print "  >> Array element recovered."

            # if j == warmingVals-1:
            #     rng = warmingVals % numCores
            #     if rng == 0:
            #         rng = numCores
            #     for index in np.arange(rng):
            #         print(">Attempting to recover array element %d, %d..." % (i, j+index-numCores))
            #         stabilityArray[i][j+index-numCores] = queue.get() # Get the unsorted array of possibly out of order results
            #         print ">>Array element recovered."

            print("Starting simulation thread for Q = %.3e, Tau_w = %.3es = %.3e yr." \
                % (Q, warmingTime, warmingTime/yrsec))
            process = multiprocessing.Process(target = simulate, args = (deltaOmega, gamma, Q, deltaT, snowballStart,\
                coolingTime, flatTime, warmingTime, queue, i, j, False,)) # Start a simulation process
            process.start() # Start the process    
            
            j += 1 # Increment j.  The placement of this matters.

            if j % numCores == 0: # Gets the values if full or finished
                for index in np.arange(numCores):
                    print("  > Attempting to recover array element %d, %d..." % (i, j+index-numCores))
                    stabilityArray[i][j+index-numCores] = queue.get() # Get the unsorted array of possibly out of order results
                    print "  >> Array element recovered."

        stabilityQaxis += ("{%d, %.1e}," % (i+1, Q))
        j = 0 # Reset index variable
        i += 1 # Increment
        print "\n"
        print("---> Simulation %.0d%% complete.\n" % int(100.0*i/Qvals)) # Give progress report

    stabilityQaxis = stabilityQaxis[:-1] + "}, None}"
    stabilityWaxis = stabilityWaxis[:-1] + "}, None}"

    print "Parsing array..."
    stabilityArray = sortClean(stabilityArray, Qvals, warmingVals) # Sort the array to make sure it is in order

    writeStabilityData(stabilityArray, stabilityQaxis, stabilityWaxis)
    raw_input("Simulation complete. %d of the %d simulations preserved resonance.\nPress enter to close this window." \
        % (np.count_nonzero(stabilityArray), Qvals*warmingVals))




def main():
    '''Main loop over multiple possible variables.'''
    print "Simulating with deltaOmega="+str(deltaOmega)+", gamma="+str(gamma)+"..."
    simulate(deltaOmega, gamma, tau, deltaT, snowballStart, coolingTime, flatTime, warmingTime, True)


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

def writeStabilityData(stabilityValues, Qaxis, Waxis):
    np.savetxt("Stability Regime.dat", stabilityValues, fmt="%s", delimiter=",", newline="},\n{")
    filehandle = open("AxesLabels.txt", "w")
    filehandle.write("FrameTicks->{"+Qaxis+","+Waxis+"}")
    filehandle.close()



#-----------------------------------------------------------------------------------------------------------------------
# Execute
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # main()
    regimeSimulation()
