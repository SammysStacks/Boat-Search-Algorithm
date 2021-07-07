"""
This simulation has been adapted from https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00sc-introduction-to-computer-science-and-programming-spring-2011/unit-2/lecture-14-sampling-and-monte-carlo-simulation/
"""

# Import Basic Modules
import sys
import math
import pylab
import random
import numpy as np
# Import Code to Simulate/Visualize the Boat's Movement
import simulateBoat
# Interpolation
from scipy import interpolate
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# Import Python Helper Files (And Their Location)
sys.path.append('./Helper Files/simulatedSource/')  # Folder with All the Helper Files
sys.path.append('./simulatedSource/')  # Folder with All the Helper Files
import extractSimulatedData

# --------------------------------------------------------------------------- #
#                            Basic Object Classes                             #
# --------------------------------------------------------------------------- #
class Position(object):
    """
    A Position represents a location in a two-dimensional tank.
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y
        
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getNewPosition(self, angle, boatSpeed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the tank.

        angle: float representing angle in degrees, 0 <= angle < 360
        boatSpeed: positive float representing boatSpeed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.getX(), self.getY()
        # Compute the change in position
        delta_y = boatSpeed * math.sin(math.radians(angle))
        delta_x = boatSpeed * math.cos(math.radians(angle))
        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        return Position(new_x, new_y)

    def __str__(self):  
        return "(%0.2f, %0.2f)" % (self.x, self.y)

class boatCollection(object):
    """
    A boatCollection represents a collection of boats inside the tank. It supports
    two operations:

    * You can add a boat to the collection with the add method.

    * You can iterate over the boats in the collection with
      "for boat in rc:". The iteration order is unspecified.
    """
    def __init__(self):
        """
        Initializes a boatCollection. The collection is initially empty.
        """
        self.boats = []
        
    def add(self, boat):
        """
        Add boat to the collection.

        boat: a boat object.
        """
        self.boats.append(boat)
        
    def __iter__(self):
        """
        Return an iterator over the boats in the collection.
        """
        return iter(self.boats)


class rectangularTank(object):
    """
    A rectangularTank represents a rectangular region containing water nodes (tiles)

    A tank has a tankWidth and a tankHeight and contains (tankWidth * tankHeight) tiles. At any
    particular time, each of these tiles are either visited or not visited
    """
    def __init__(self, tankWidth, tankHeight):
        """
        Initializes a rectangular tank with the specified tankWidth and tankHeight.

        Initially, no tiles in the tank have been visited.

        tankWidth: an integer > 0
        tankHeight: an integer > 0
        """
        # Define Basic Parameters
        self.tankWidth = int(tankWidth)
        self.tankHeight = int(tankHeight)
        self.tiles = {}
        
        # Initialize the Board
        self.initializeBoard()
    
    def initializeBoard(self):
        for x in range(self.tankWidth):
            for y in range(self.tankHeight):
                self.tiles[(x, y)] = False
            
    def markAsVisited(self, pos):
        """
        Mark the tile under the position POS as visited.
        Assumes that POS represents a valid position inside this tank.

        pos: a Position object
        """
        x = math.floor(pos.getX())
        y = math.floor(pos.getY())
        self.tiles[(x, y)] = True
        
    def hasVisited(self, m, n):
        """
        Return True if the tile (m, n) has been visited.
        Assumes that (m, n) represents a valid tile inside the tank.

        m: an integer
        n: an integer
        returns: True if (m, n) was visited, False otherwise
        """
        return self.tiles[(m, n)]
    
    def getNumTiles(self):
        """
        Return the total number of tiles in the tank.

        returns: an integer
        """
        return self.tankWidth * self.tankHeight
    
    def getNumVisitedTiles(self):
        """
        Return the total number of visited tiles in the tank.

        returns: an integer
        """
        return sum(self.tiles.values())
    
    def getRandomPosition(self):
        """
        Return a random position inside the tank.

        returns: a Position object.
        """
        return Position(random.random() * self.tankWidth,
                        random.random() * self.tankHeight)
    
    def isPositionIntank(self, pos):
        """
        Return True if pos is inside the tank.

        pos: a Position object.
        returns: True if pos is in the tank, False otherwise.
        """
        return ((0 <= pos.getX() < self.tankWidth)
                and (0 <= pos.getY() < self.tankHeight))


class cosmolSimTank(rectangularTank):
        
    def __init__(self, sourceLocations, tankWidth, tankHeight, simFile):
        super().__init__(tankWidth, tankHeight)  # Get Variables Inherited from the helper_Files Class
        
        self.mapedTiles = {}        
        self.getSimData(simFile)
        
        # Initialize the Board
        self.initializeBoard()
    
    def dataRound(self, array):
        return np.round(array, 5)
        
    def getSimData(self, simFile):
        # Extract the Data from the Excel File
        self.simX, self.simY, self.simZ = extractSimulatedData.processData().getData(simFile)
        # Shift to Start at Zero,Zero
        self.simX -= min(self.simX)
        self.simY -= min(self.simY)
        # Round X,Y so its Discrete and Comparable
        self.simX = self.dataRound(self.simX)
        self.simY = self.dataRound(self.simY)
        # Respecify the Single Source
        maxIndex = np.argmax(self.simZ)
        self.sourceLocations = [(self.simX[maxIndex], self.simY[maxIndex])]
        
        # Reshape Board to Match Simulation
        self.tankWidth = int(max(self.simX))
        self.tankHeight = int(max(self.simY))
        
        # Save Interpolated Form
        self.interpolateSim = interpolate.interp2d(self.simX, self.simY, self.simZ)
        
        # Store Data in Mapped Tiles Data Structure
        positions = list(zip(self.simX, self.simY))
        self.mapedTiles = dict(zip(positions, self.simZ))
        
        # Reinitialize Tiles
        self.tiles = dict(zip(positions, len(self.simZ)*[False]))
        self.initializeBoard()
        self.plotSimData()
    
    def plotSimData(self):        
        # Plot Model
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.simX, self.simY, self.simZ, c=self.simZ)
        plt.show()
        
        fig = plt.figure()
        plt.scatter(self.simX, self.simY, c=self.simZ)
        plt.show()
    
    def posReading(self, currentPos):
        # self.interpolateSim(currentPos[0], currentPos[1])[0]
        return self.mapedTiles.get(currentPos) or self.mapedTiles[ 
                min(self.mapedTiles.keys(), key = lambda key: self.euclideanDist(key, currentPos))]
    
    def euclideanDist(self, P1, P2):
        return np.linalg.norm((P1[0]-P2[0], P1[1]-P2[1]))
    
    def sourceFound(self):
        for sourceLocation in self.sourceLocations:
            if self.tiles[sourceLocation] == True:
                return True
        return False


class diffusionModelTank(rectangularTank):
    
    def __init__(self, sourceLocations, tankWidth, tankHeight):
        super().__init__(tankWidth, tankHeight)  # Get Variables Inherited from the helper_Files Class
        
        self.mapedTiles = {}
        self.sourceLocations = sourceLocations
        self.scaleTiles = 10
        
        #self.diffuseSources()
    
    def initializeMap(self):
        for x in range(self.tankWidth*self.scaleTiles):
            x = x/self.scaleTiles
            for y in range(self.tankHeight*self.scaleTiles):
                y = y/self.scaleTiles
                self.mapedTiles[(x, y)] = 0
    
    def diffuseModel(self, delX, delY):
        return 10*math.exp(-(delY**2 + delX**2))
    
    def diffuseSources(self):
        self.initializeMap()
        
        for sourceLocation in self.sourceLocations:
            xSource = sourceLocation[0]
            ySource = sourceLocation[1]
            for x in range(self.tankWidth):
                delX = x - xSource
                for y in range(self.tankHeight):
                    delY = y - ySource
                    
                    self.mapedTiles[(x, y)] += self.diffuseModel(delX, delY)
    
    def plotDiffuseModel(self):
        # Unpack Tuples
        xy, z = zip(*self.mapedTiles)
        x,y = zip(*xy)
        
        # Plot Model
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x,y,z,c=z)
    
    def posReading(self, currentPos):
        # Loop Through Each Source to Find its Contribution
        sensorReading = 0
        for sourceLocation in self.sourceLocations:
            xSource = sourceLocation[0]
            ySource = sourceLocation[1]
            
            delX = currentPos[0] - xSource
            delY = currentPos[1] - ySource
                    
            sensorReading += self.diffuseModel(delX, delY)
        
        return sensorReading
    
    def sourceFound(self):
        for sourceLocation in self.sourceLocations:
            if self.tiles[sourceLocation] == True:
                return True
        return False


class Boat(object):
    """
    Represents a boat finding the source

    At all times the boat has a particular position and direction in the tank.
    The boat also has a fixed speed.

    Subclasses of boat should provide movement strategies by implementing
    updatePosition(), which simulates a single time-step.
    """
    def __init__(self, tank, boatSpeed, boatLocation = Position(0,0)):
        """
        Initializes a boat with the given speed in the specified tank. The
        boat initially has a random direction and a random position in the
        tank. 

        tank:  a rectangularTank object.
        boatSpeed: a float (boatSpeed > 0)
        """
        # Initialize Boat Parameters
        self.boatSpeed = boatSpeed
        self.direction = random.randrange(360)
        self.position = Position(boatLocation[0], boatLocation[1])
        
        # Initialize Place in Tank
        self.tank = tank
        self.tank.markAsVisited(self.position)
        
        # Initialize Sensor Parameters
        self.sensorAngle = 30;
        self.sensorDistance = 1.5;
        
        # Keep Track of Past Movements
        self.pastValues = {}
        
    def getBoatPosition(self):
        """
        Return the position of the boat.

        returns: a Position object giving the boat's position.
        """
        return self.position
    
    def getBoatDirection(self):
        """
        Return the direction of the boat.

        returns: an integer d giving the direction of the boat as an angle in
        degrees, 0 <= d < 360.
        """
        return self.direction
    
    def setBoatPosition(self, position):
        """
        Set the position of the boat to POSITION.

        position: a Position object.
        """
        self.position = position
        
    def setBoatDirection(self, direction):
        """
        Set the direction of the boat to DIRECTION.

        direction: integer representing an angle in degrees
        """
        self.direction = direction
    
    def updatePastRecord(self, currentPosObj, currentVal):
        # Get the Current Position (Rounded to Current Square)
        x = math.floor(currentPosObj.getX())
        y = math.floor(currentPosObj.getY())
        currentPos = (x,y)
        # Get Previous Past Values at the Position
        pastVals = self.pastValues.get(currentPos, [])
        # Add the New Position's Value
        pastVals.append(currentVal)
        # Store the New Value in the Dictionary if No Position was There
        if len(pastVals) == 1:
            self.pastValues[currentPos] = pastVals
    
    def getSensorsPos(self, currentPosObj):
        # Get the Current Position
        x = currentPosObj.getX()
        y = currentPosObj.getY()
        # Find Location of the Front Sensor
        sensorFrontX = x;
        sensorFrontY = y + self.sensorDistance
        # Find the Location of the Left Sensor
        sensorLeftX = x + self.sensorDistance*math.cos(math.radians(180 + self.sensorAngle)); 
        sensorLeftY = y + self.sensorDistance*math.sin(math.radians(180 + self.sensorAngle)); 
        # Find the Location of the Right Sensor
        sensorRightX = x + self.sensorDistance*math.cos(math.radians(-self.sensorAngle)); 
        sensorRightY = y + self.sensorDistance*math.sin(math.radians(-self.sensorAngle)); 
        # Return the Three Sensor Positions
        return (sensorFrontX, sensorFrontY), (sensorLeftX, sensorLeftY), (sensorRightX, sensorRightY)
    


# --------------------------------------------------------------------------- #
#                  Movement Strategies: Search Algorythms                     #
# --------------------------------------------------------------------------- #

class standardBoat(Boat):
    """
    A Standardboat is a boat with the standard movement strategy.

    At each time-step, a Standardboat attempts to move in its current direction; when
    it hits a wall, it chooses a new direction randomly.
    """
    
    def __init__(self, tank, boatSpeed, boatLocations):
        super().__init__(tank, boatSpeed, boatLocations)
        
    def updatePosition(self):
        """
        Simulate the passage of a single time-step.

        Move the boat to a new position and mark the tile it is on as having
        been Visited.
        """
        candidatePosition = self.position.getNewPosition(self.direction, self.boatSpeed)
        if self.tank.isPositionIntank(candidatePosition):
            self.setBoatPosition(candidatePosition)
            self.tank.markAsVisited(self.position)
        else:
            self.direction = random.randrange(360)
            
            
class randomBoat(Boat):
    """
    A randomBoat is a boat with the random movement strategy.

    At each time-step, a randomBoat picks a direction and angle and moves there
    """
    
    def __init__(self, tank, boatSpeed, boatLocations):
        super().__init__(tank, boatSpeed, boatLocations)
        
    def updatePosition(self):
        """
        Simulate the passage of a single time-step.

        Move the boat to a new position and mark the tile it is on as having
        been Visited.
        """
        cur_pos = self.getBoatPosition()
        cur_dir = self.getBoatDirection()
        self.setBoatDirection(random.randrange(360))
        new_pos = cur_pos.getNewPosition(cur_dir, self.boatSpeed)
        if self.tank.isPositionIntank(new_pos):
            self.setBoatPosition(new_pos)
            self.tank.markAsVisited(new_pos)

class gradientDecent(Boat):
    """
    Move to the Highest Gradient
    """
    
    def __init__(self, tank, boatSpeed, boatLocations):
        super().__init__(tank, boatSpeed, boatLocations)
        
    def updatePosition(self):
        """
        Simulate the passage of a single time-step.

        Move the boat to a new position and mark the tile it is on as having
        been Visited.
        """
        # Find Your Current Location
        currentPos = self.position
        # Find the Location of Each of the Three Sensors
        frontSensorPos, leftSensorPos, rightSensorPos = self.getSensorsPos(currentPos)
        # Find the Interpolated Values at the Sensor's Position
        frontSensorVal = self.tank.posReading(frontSensorPos)
        leftSensorPosVal = self.tank.posReading(leftSensorPos)
        rightSensorPosVal = self.tank.posReading(rightSensorPos)
        
        # Define Each Sensor in 3D Space
        frontPoint = np.array([frontSensorPos[0], frontSensorPos[1], frontSensorVal])
        leftPoint = np.array([leftSensorPos[0], leftSensorPos[1], leftSensorPosVal])
        rightPoint = np.array([rightSensorPos[0], rightSensorPos[1], rightSensorPosVal])
                
        # Find the Normal Vector to the 3-Point Plane
        gradSensor = np.cross(frontPoint - leftPoint, rightPoint - leftPoint)
        if np.linalg.norm(gradSensor) != 0:
            gradSensor = gradSensor*(2*(gradSensor[2] < 0) - 1)
            # Make Tangent Plane
            #d = np.dot(gradSensor, frontPoint)
            #gradSensor /= -d
            #print('The equation is {0}x + {1}y + {2}z = {3}'.format(gradSensor[0], gradSensor[1], gradSensor[2], -1))
            
            # Get the Direction of Max Increase
            newDirection = [gradSensor[0], gradSensor[1]]
            if np.linalg.norm(newDirection) == 0:
                allPoints = [frontPoint, leftPoint, rightPoint]
                newDirection = max(allPoints, key=lambda x:x[-1])[0:2]

            # Get New Angle
            referenceAngle = [1,0]
            unitVectorDirection = newDirection/np.linalg.norm(newDirection)
            dot_product = np.dot(unitVectorDirection, referenceAngle)
            newAngle = np.degrees(np.arccos(dot_product))
            # Correction Factor to Angle as Cosine Repeats Values
            needCorrection = newDirection[1] < 0
            if needCorrection:
                newAngle = 360 - newAngle
        else:
            print("Gradient is Zero")
            newAngle = random.randrange(360)
        
        # Check to See if the Position is in the Tank
        candidatePosition = currentPos.getNewPosition(newAngle, self.boatSpeed)
        while not self.tank.isPositionIntank(candidatePosition):
            candidatePosition = currentPos.getNewPosition(random.randrange(360), self.boatSpeed*2)
        
        #self.updatePastRecord(candidatePosition, self.tank.posReading(candidatePosition))
        #if self.pastValues()
        
        # Move to the Position
        self.setBoatDirection(newAngle)
        self.setBoatPosition(candidatePosition)
        self.tank.markAsVisited(self.position)
                        


# --------------------------------------------------------------------------- #
#                             Run Boat Simulation                             #
# --------------------------------------------------------------------------- #

def runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats = 1, simFile = "./", visualize = True):
    """
    Runs NUM_TRIALS trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction MIN_COVERAGE of the tank.

    The simulation is run with numBoats boats of type boat_TYPE, each with
    speed SPEED, in a tank of dimensions tankWidth x tankHeight.

    sourceLocations: A list of tuples representing the x,y locations of the sources [(1,2),(3,4)]
    boatLocations: A list of tuples representing the x,y locations of the boats [(1,2),(3,4)]
    boatSpeed: a float (boatSpeed > 0)
    tankWidth: an int (tankWidth > 0)
    tankHeight: an int (tankHeight > 0)
    numBoats: an int (numBoats > 0)
    visualize: Boolean
    """
    # Initialize the Tank
    waterTank = cosmolSimTank(sourceLocations, tankWidth, tankHeight, simFile)
    #waterTank = diffusionModelTank(sourceLocations, tankWidth, tankHeight)
    boatType = gradientDecent
    # Initialize Evaluation Oarameters
    total_time_steps = 0.0
    # Initialize Animation for Searching
    if visualize:
        anim = simulateBoat.boatVisualization(numBoats, waterTank.tankWidth, waterTank.tankHeight)
    
    # Add the Boats to the Tank
    boatCollection = []
    for boatNum in range(numBoats):
        boatCollection.append(boatType(waterTank, boatSpeed, boatLocations[boatNum]))
    if visualize:
        anim.update(waterTank, boatCollection)
    
    # Run the Search Algorythm Until the Boat Reaches the Source
    while not waterTank.sourceFound():
        # Move Each Boat
        for boat in boatCollection:
            boat.updatePosition()
        total_time_steps += 1
        # Update Animation with the Movement
        if visualize:
            anim.update(waterTank, boatCollection)
    if visualize:
        anim.done()
    
    #Return the Total Time Steps it Took
    return total_time_steps



def showPlot1(title, x_label, y_label, sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats):
    """
    Produces a plot comparing the two boat strategies in a 20x20 tank with 80%
    minimum coverage.
    """
    num_boat_range = range(1, 11)
    times1 = []
    times2 = []
    for numBoats in num_boat_range:
        print("Plotting", numBoats, "boats...")        
        times1.append(runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats))
        times2.append(runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats))
    pylab.plot(num_boat_range, times1) 
    pylab.plot(num_boat_range, times2)
    pylab.title(title)
    pylab.legend(('Standardboat', 'RandomWalkboat'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

    
def showPlot2(title, x_label, y_label, sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats):
    """
    Produces a plot showing dependence of cleaning time on tank shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    for tankWidth in [10, 20, 25, 50]:
        tankHeight = 300/tankWidth
        print("Plotting cleaning time for a tank of Width:", tankWidth, "by Height:", tankHeight)
        aspect_ratios.append(float(tankWidth) / tankHeight)
        times1.append(runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats))
        times2.append(runSimulation(sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats))
    pylab.plot(aspect_ratios, times1)
    pylab.plot(aspect_ratios, times2)
    pylab.title(title)
    pylab.legend(('Standardboat', 'RandomWalkboat'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()


if __name__ == '__main__':
    sourceLocations = [(5,6)]
    boatLocations = [(0,0)]
    boatSpeed = 2
    tankWidth = 20
    tankHeight = 20
    numBoats = 1
    
    showPlot1('Time to clean 80% of a 20x20 tank, for various numbers of boats', 'Number of boats',
              'Time/ steps', sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats)
    showPlot2('Time to clean 80% of a 400-tile tank for various tank shapes', 'Aspect Ratio',
              'Time / steps', sourceLocations, boatLocations, boatSpeed, tankWidth, tankHeight, numBoats)
    
    
    
