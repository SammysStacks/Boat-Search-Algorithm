"""
    Written by Samuel Solomon
    This simulation has been adapted from https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00sc-introduction-to-computer-science-and-programming-spring-2011/unit-2/lecture-14-sampling-and-monte-carlo-simulation/
    
    --------------------------------------------------------------------------
    Program Description:
    
    Model the Boat Searching for the Chemical Source in Water
    Assumes the Boat has Three Sensors to Read Data
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        %conda install matplotlib
        %conda install numpy
        %pip install pyexcel
        %pip install scipy
        
    --------------------------------------------------------------------------
"""

# Basic Modules
import sys
# Import Python Helper Files (And Their Location)
sys.path.append('./Helper Files/')  # Folder with All the Helper Files
sys.path.append('./Helper Files/simulatedSource/')  # Folder with All the Helper Files
# Import Helper Files
import objectParameters


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Specify the Boat Parameters
    numBoats = 1   # The Number of Boats Checking for the Source.
    boatSpeed = 2 # numtiles/movement. Can Move at an Arbitrary Angle. Units: cm
    boatLocations = [(30,35)]  # (2,5) with 2.4 @30x30;  (33,22) with 2.02 @35x35
    boatDirection = [1,1] # The Initial Direction of the Boat 
    sensorDistance = 1.6 # Distance from the Boat's Sensor to the Boat's Center

    # Specify the Tank's Parameters
    tankWidth = 40 # The width of the tank (Inches). Must be an Integer
    tankHeight = 40 # The height of the tank (Inches). Must be an Integer
    
    # Specify the Source Locations
    sourceLocations = [(20, 20), (15,27)]
    # Specify the Simulation Data
    simFile = './Helper Files/simulatedSource/Input Data/Excel Files/diffusion_two_drop_4M_0speed_2.xlsx'
    
    # ---------------------------------------------------------------------- #
    #                        Running Boat Simulation                         #
    # ---------------------------------------------------------------------- #

    #searchObj = objectParameters.runSimulation(sourceLocations, boatLocations, boatSpeed, boatDirection, tankWidth, tankHeight, numBoats, simFile)
    points = []
    for x in range(41):
        for y in range(41):
            points.append((x,y))
    for point in points:
        x = point[0]; y = point[1]
        boatLocations = [point]
        
        outFile = "./ALL/AStar_" + str(x) + "-" + str(y) + ".png"
        # algPositions, fullData = objectParameters.runSimulation(sourceLocations, boatLocations, boatSpeed, boatDirection, sensorDistance, tankWidth, tankHeight, numBoats, simFile, True)
        algPositions, fullData = objectParameters.compareAlgorythms(sourceLocations, boatLocations, boatSpeed, boatDirection, sensorDistance, tankWidth, tankHeight, numBoats, simFile, outFile)
    
    