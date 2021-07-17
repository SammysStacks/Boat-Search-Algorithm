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

import objectParameters


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Specify the Boat Parameters
    numBoats = 1   # The Number of Boats Checking for the Source.
    boatSpeed = 1  # numtiles/movement. Can Move at an Arbitrary Angle
    boatLocations = [(10,3)]

    # Specify the Tank's Parameters
    tankWidth = 30  # The width of the tank (Inches). Must be an Integer
    tankHeight = 30 # The height of the tank (Inches). Must be an Integer
    
    # Specify the Source Locations
    sourceLocations = [(17, 29)]
    # Specify the Simulation Data
    simFile = './Helper Files/simulatedSource/Input Data/diffusion4.xlsx'
    
    # ---------------------------------------------------------------------- #
    #                        Running Boat Simulation                         #
    # ---------------------------------------------------------------------- #

    searchObj = objectParameters.runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats, simFile)
