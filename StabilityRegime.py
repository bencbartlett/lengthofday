# Analysis of a Snowball Earth Model in Breaking Atmospheric Resonance
# Ben Bartlett - Ph11 Research Project
# Winter 2014 - Summer 2014

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Importations
#-----------------------------------------------------------------------------------------------------------------------

from math import pi                 # Cause I get tired of typing np.pi over and over and over agin...
import numpy as np                  # We kind of need this
import matplotlib.pyplot as plt     # Oooh, pretty graphs!
import multiprocessing              # For super-duper-mega-speedarific(TM) computations
import time                         # So we can know what's going on when


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
Fnaught = 4.7*10**13 / 5.1*10**14   # Heat flux in W/m^2, from http://www.solid-earth.net/1/5/2010/se-1-5-2010.pdf
I = np.float64(8.0*10**37)          # Moment of Inertia of Earth
hnaught = 28644                     # Meters, just played around in wolfram alpha to find a value that works
omeganaught = (g*hnaught)**.5/Rearth# Natural resonance frequency of earth's atmosphere, comes out to 2*pi/21hr
Tnaught = 287                       # Surface temperature in K

# Lunar torque scaling
moonT = np.float64(2.3 * 10**20)    # Precambrian lunar torque was 2.3E23 dyne-cm = 2.3E20 NM, according to Zahnle paper
                                    # Scaling factor for atmospheric torque


# Adjustable Simulation Parameters:-------------------------------------------------------------------------------------

# Sinusoidal Noise:
numSines = 0
gamma = 1*10**7 * yrsec             # Frequency modifier for Omega0 = omega0 + deltaOmega * sin(2pi/gamma*t)
deltaOmega = 0*2*pi/(21*3600*20)    # Amplitude modifier for Omega0 = omega00 + deltaOmega * sin(2pi/gamma*t)


# Snowball Earth Variables
deltaT = 25                         # Temperature change in K for snowball earth
snowballStart = .1*10**9 * yrsec    # When the snowball earth starts
coolingTime = .2*10**9 * yrsec      # Time it takes to cool down by deltaT
flatTime = .1*10**9 * yrsec         # How long it remains at cooler temperature
# warmingTime = 1*10**7 * yrsec       # Time it takes to warm back up
# coolingSlope = -deltaT/coolingTime
# warmingSlope = deltaT/warmingTime


# Time and Initial Value Parameters
tStep = 100 * yrsec                 # Step size in seconds for time variable
                                    # Note that at the moment, a small step size is required for accurate calculations.
tmax = 0.6*10**9 * yrsec            # Simulation stops when it reaches this time value
#omegastart = 2*pi/(21.06*3600)      # 2pi/5hr - Initial LoD of Earth
omegastart = 2*pi/(15*3600)


# Miscellaneous Parameters
numCores = multiprocessing.cpu_count() # Gets the number of cores on the computer to optimize number of processes
variance = 1.025                    # Error bounds.  To be accepted as stable, 1/variance < omega/(2pi/21) < variance

# Resolution for Q by Tau graph:
Qvals = 8
warmingVals = 8
tempVals = 1
TevaluationValues = [25]
QevaluationValues = np.linspace(300, 30, Qvals)
WevaluationValues = np.logspace(np.log10(1000 * yrsec), np.log10(1*10**8 * yrsec), warmingVals)



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
    return -1* moonT * (omega/(2*pi/(24*3600)))**6                  # This should eventually be replaced by a 1/r^6 term

def atmTorque(omega, t, tau, A24, gamma, deltaOmega, T, h, coolingTime, \
    coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp):
    '''Returns atmospheric torque following the analytic solution that was solved for.'''
    omega0, newT, newh = resonance(t, deltaOmega, gamma, T, h, coolingTime, coolingSlope, \
        flatTime, warmingTime, warmingSlope, freq, phi, amp)
    # return -Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - omega0**2) + omega/(tau**2))\
    #     / (4*(omega**2 - omega0**2)**2 + (omega**2)/(tau**2)), newT, newh
    #return -1/20 * moonT * (Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - omega0**2) + omega/(tau**2))\
    #    / (4*(omega**2 - omega0**2)**2 + (omega**2)/(tau**2))) / A24, newT, newh
    return moonT/20 * (Fnaught/(2*rho*Cp*T) * (4*omega*(omega**2 - omega0**2) + omega/(tau**2))\
        / (4*(omega**2 - omega0**2)**2 + (omega**2)/(tau**2))) / A24, newT, newh

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

    lowerTimeLim = 15
    upperTimeLim = 24

    for dayLength in np.linspace(lowerTimeLim, upperTimeLim, 1000):
        omega = (2*pi)/(3600*dayLength)
        moonTorques.append(-moonTorque(omega))
        atmTorques.append(atmTorque(omega, t, tau, A24, gamma, deltaOmega, T, h, coolingTime, \
            coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)[0])

    plt.xlim([lowerTimeLim, upperTimeLim])
    plt.plot(np.linspace(lowerTimeLim, upperTimeLim, 1000), atmTorques, label = "Atmospheric torque")
    plt.plot(np.linspace(lowerTimeLim, upperTimeLim, 1000), moonTorques, label = "-1 * Lunar torque")
    plt.xlabel("Length of Day (hr)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.show()

def sortClean(array, xdim, ydim): # Edit: tested and confirmed working
    '''Sorts the processing results to make sure they are in order and cleans up the data by removing indices.'''
    newarray = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            newarray[array[i][j][1]][array[i][j][2]] = array[i][j][0] # Sort the arrays by indices
    return newarray


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
    np.savetxt("Stability Regime Copypaste.dat", stabilityValues, fmt="%s", delimiter=",", newline="},\n{")
    np.savetxt('StabilityRegime.dat', stabilityValues, fmt = '%i', delimiter = ',') # Exports CSV format
    filehandle = open("AxesLabels.txt", "w")
    filehandle.write("FrameTicks->{"+Qaxis+","+Waxis+"}")
    filehandle.close()

def showStabilityData(stabilityValues, Qaxis, Waxis):
    '''Displays a pretty plot :P'''
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

def simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, queue, i, j, \
    printTrue, plotTrue, putToQueue):
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

    # Current atmospheric column height
    A24 = Fnaught/(2*rho*Cp*Tnaught) * (4*2*pi/(24*3600)*(2*pi/(24*3600)**2 - omeganaught**2) + \
        2*pi/(24*3600)/(tau**2)) / (4*(2*pi/(24*3600)**2 - omeganaught**2)**2 + (2*pi/(24*3600)**2)/(tau**2))
    # plotSineNoise()

    # print tau, deltaT, snowballStart, coolingTime, flatTime, warmingTime, queue, i, j, plotTrue
    
    counter = 0
    while t <= tmax:
        if counter == 1000 and printTrue:
            print("Time: %.3f Myr   Omega: %.10f   Day length: %.10f   Temperature: %.10f"\
                % ((t/(yrsec*1000000)), omega, 2*pi/(3600*omega), T-273))
            counter = 0

        atmTorqueReults, newT, newh = atmTorque(omega, t, tau, A24, gamma, deltaOmega, T, h, coolingTime, \
            coolingSlope, flatTime, warmingTime, warmingSlope, freq, phi, amp)
        domega = (atmTorqueReults + moonTorque(omega)) / I * tStep 
        omega += domega                                             # Increment omega
        T, h = newT, newh                                           # Update temperature and column height values

        dayLengthValues.append(2*pi/(3600*omega))                   # Store day length
        omegaValues.append(omega)
        tempValues.append(T - 273)

        t += tStep                                                  # Increment 
        counter += 1
    
    if plotTrue:
        plot(dayLengthValues, tempValues)

    if putToQueue:
        queue.put([isStable(omegaValues[-1]), i, j])                # For multiprocessing, put the return value to queue

    return isStable(omegaValues[-1])                                # Get last omega value of simulation

def testSimulate(value):
    if value >= 505:
        return True
    else:
        return False

def binarySearchRegime(Q, deltaT, Qindex, Tindex, queue, printTrue = False, putToQueue = False):
    '''Same thing as the regimeSimulation() function, except instead of simulating every value; it searches for
    the boundary of the stability-nonstability region by iterating over warmingTime for a given Q and deltaT.'''
    lowerBound = 0
    upperBound = warmingVals - 1
    wPosition = int(np.mean([lowerBound, upperBound])) # Start at the middle of the w-values array
    while True:
        # Simulate the function to see if it's stable or not

        result = simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, \
            WevaluationValues[wPosition], queue, Qindex, Tindex, \
            printTrue = False, plotTrue = False, putToQueue = False)
        # print wPosition
        # result = testSimulate(wPosition)
        if result == True:  # If simulation is stable, everything right of it ("above" it) is stable
            upperBound = wPosition
            if wPosition == int(np.mean([lowerBound, upperBound])): # If no change in position
                if putToQueue:
                    queue.put([Qindex, wPosition, Tindex])
                return wPosition   # Returns the last stable value
            wPosition = int(np.mean([lowerBound, upperBound]))

        else:  # If simulation is not stable, everything "below" it is not stable
            lowerBound = wPosition
            if wPosition == int(np.mean([lowerBound, upperBound])):  # If no change in position
                if putToQueue:
                    queue.put([Qindex, wPosition, Tindex])
                return wPosition  # Returns the last stable value
            wPosition = int(np.mean([lowerBound, upperBound]))


#-----------------------------------------------------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------------------------------------------------

def regimeSimulation():
    '''Simulates through a large number of variable combinations to find an area of stability-preserving conditions.'''
    queue = multiprocessing.Queue() # Initialize a multiprocessing queue for communication between processes
    startTime = time.clock()

    stabilityArray = np.zeros((Qvals, warmingVals, tempVals)) # Initial unsorted array output
    i = j = k = currentProcesses = 0 # Counter variables to track position
    iPos = jPos = 0 # Position variables to track array placement

    stabilityQaxis = "{{" # These strings track the axes for easy input into Mathematica

    for Q in QevaluationValues:
        # Start the binary search fo a given Q, T pair
        print("Starting binary search for Q = %.3e, deltaT = %.3eK." \
                % (Q, deltaT))
        # Start a simulation process
        process = multiprocessing.Process(target = binarySearchRegime, args = (Q, deltaT, i, k, queue, True, True,))
        process.start() # Start the process 
        currentProcesses += 1 # Increment number of processes

        # Retrieve results
        if currentProcesses == numCores or i == Qvals - 1: # Gets the values if full or finished
            recoveryAttempts = min(currentProcesses, (Qvals - iPos))
            print "\n"
            for index in np.arange(recoveryAttempts):
                print("  > Attempting to recover array row %d..." % iPos)
                result = queue.get() # Get the unsorted array of unordered results
                stabilityArray[result[0], result[1]:, result[2]] = 1 # Sets all w values above boundary to stable
                print ("  >> Array element recovered for (Q,dT) = (%d,%d). Set all tw values above %d to stable." \
                    % (result[0], result[2], result[1]))
                iPos += 1
            currentProcesses = 0 # Reset number of processes
            print "\n"

        stabilityWaxis = "{{"

        for warmingTime in WevaluationValues:
            if j % 8 == 0:
                stabilityWaxis += ("{%d, %.1e}," % (j+1, warmingTime/yrsec))

            # print("Starting simulation thread for Q = %.3e, Tau_w = %.3es = %.3e yr." \
            #     % (Q, warmingTime, warmingTime/yrsec))
            # process = multiprocessing.Process(target = simulate, args = (deltaOmega, gamma, Q, deltaT, snowballStart,\
            #     coolingTime, flatTime, warmingTime, queue, i, j, False, False, True,)) # Start a simulation process
            # process.start() # Start the process    
            
            # j += 1 # Increment j.  The placement of this matters.
            # currentProcesses += 1 # Increment number of processes

            # if currentProcesses == numCores or (i==Qvals-1 and j==warmingVals-1): # Gets the values if full or finished
            #     recoveryAttempts = min(currentProcesses, (Qvals*warmingVals - (iPos*warmingVals + jPos)) )
            #     for index in np.arange(recoveryAttempts):
            #         print("  > Attempting to recover array element (%d,%d)..." % (iPos, jPos))
            #         stabilityArray[iPos][jPos] = queue.get() # Get the unsorted array of unordered results
            #         print "  >> Array element recovered."
            #         jPos += 1
            #         if jPos > warmingVals - 1:
            #             jPos = 0
            #             iPos += 1
            #     currentProcesses = 0 # Reset number of processes

        if i % 4 == 0:
            stabilityQaxis += ("{%d, %.1e}," % (i+1, Q))
        j = 0 # Reset index variable
        i += 1 # Increment
        print "\n"

        timeEstimate = (1-i/Qvals)*(Qvals/i)*(time.clock() - startTime)/(3600)

        print("---> Simulation %.0d%% complete. Estimated time remaining: %.2f hr.\n" % \
            (int(100.0*i/Qvals),  timeEstimate)) # Give progress report

    stabilityQaxis = stabilityQaxis[:-1] + "}, None}"
    stabilityWaxis = stabilityWaxis[:-1] + "}, None}"

    print "Parsing array..."
    # stabilityArray = sortClean(stabilityArray, Qvals, warmingVals) # Sort the array to make sure it is in order

    writeStabilityData(stabilityArray, stabilityQaxis, stabilityWaxis)
    # showStabilityData(stabilityArray, QevaluationValues, WevaluationValues)
    raw_input("Simulation complete. %d of the %d simulations preserved resonance.\nPress enter to close this window." \
        % (np.count_nonzero(stabilityArray), Qvals*warmingVals))


def singleSimulation():
    '''Main loop over multiple possible variables.'''
    print "Simulating with deltaOmega=" + str(deltaOmega) + ", gamma=" + str(gamma) + "..."
    queue = multiprocessing.Queue()
    Q = 100                             # Q factor of the atmosphere
    warmingTime = 1 * 10**7 * yrsec
    i=j=0
    # plotTorque(Q)
    simulate(deltaOmega, gamma, Q, deltaT, snowballStart, coolingTime, flatTime, warmingTime, queue, i, j, \
        printTrue = True, plotTrue = True, putToQueue = False)


#-----------------------------------------------------------------------------------------------------------------------
# Execute
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    singleSimulation()
    #regimeSimulation()
    #binarySearchRegime(100, 25)

