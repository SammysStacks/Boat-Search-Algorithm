"""
This simulation has been adapted from https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00sc-introduction-to-computer-science-and-programming-spring-2011/unit-2/lecture-14-sampling-and-monte-carlo-simulation/
"""

# Import Basic Modules
import math
import time
# Modules to Interact with Application 
import tkinter as tk
#from PIL import Image, ImageTk

class boatVisualization:
    def __init__(self, numBoats, width, height, delay = 0.2):
        "Initializes a visualization with the specified parameters."
        # Number of seconds to pause after each frame
        self.delay = delay

        self.max_dim = max(width, height)
        self.width = width
        self.height = height
        self.numBoats = numBoats

        # Initialize a drawing surface
        self.master = tk.Tk()
        self.w = tk.Canvas(self.master, width=500, height=500)
        self.w.pack()
        self.master.update()

        # Draw a backing and lines
        x1, y1 = self._map_coords(0, 0)
        x2, y2 = self._map_coords(width, height)
        self.w.create_rectangle(x1, y1, x2, y2, fill = "#03a9f4", width = 3)

        # Draw gray squares for dirty tiles
        self.tiles = {}
        for i in range(width):
            for j in range(height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)
                self.tiles[(i, j)] = self.w.create_rectangle(x1, y1, x2, y2,
                                                             fill = "#242546", dash=(1,1))

        # Draw some status text
        self.boats = None
        self.text = self.w.create_text(25, 0, anchor=tk.NW,
                                       text=self._status_string(0, 0))
        #self.boatImage = ImageTk.PhotoImage(file="./Helper Files/Boat Images/boat copy.jpg")
        
        self.time = 0
        self.master.update()

    def _status_string(self, time, num_visited_tiles):
        "Returns an appropriate status string to print."
        percent_visited = 100 * num_visited_tiles / (self.width * self.height)
        return "Time: %04d; %d tiles (%d%%) visited" % \
            (time, num_visited_tiles, percent_visited)

    def _map_coords(self, x, y):
        "Maps grid positions to window positions (in pixels)."
        return (250 + 450 * ((x - self.width / 2.0) / self.max_dim),
                250 + 450 * ((self.height / 2.0 - y) / self.max_dim))

    def _draw_boat(self, position, direction):
        "Returns a polygon representing a boat with the specified parameters."
        x, y = position.getX(), position.getY()
        x1, y1 = self._map_coords(x, y)
        x2, y2 = self._map_coords(x + 1 * math.cos(math.radians(direction)),
                          y + 1 * math.sin(math.radians(direction)))
        
        return self.w.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill = '#380000')

    def update(self, tank, boats):
        "Redraws the visualization with the specified tank and boat state."
        # Removes a gray square for any tiles have been visiteded.
        for i in range(self.width):
            for j in range(self.height):
                if tank.hasVisited(i, j):
                    self.w.delete(self.tiles[(i, j)])
        # Delete all existing boats.
        if self.boats:
            for boat in self.boats:
                self.w.delete(boat)
                self.master.update_idletasks()
        # Draw new boats
        self.boats = []
        for boat in boats:
            pos = boat.getBoatPosition()
            x, y = pos.getX(), pos.getY()
            x1, y1 = self._map_coords(x - 0.08, y - 0.08)
            x2, y2 = self._map_coords(x + 0.08, y + 0.08)
            #self.boats.append(self.w.create_image(x,y, image = self.boatImage))
            self.boats.append(self.w.create_oval(x1, y1, x2, y2, fill = "black"))
            self.boats.append(
                self._draw_boat(boat.getBoatPosition(), boat.getBoatDirection()))
        # Add Source
        for sourceLocation in tank.sourceLocations:
            x, y = sourceLocation[0] + 0.5, sourceLocation[1] + 0.5
            x1, y1 = self._map_coords(x - 0.1, y - 0.1)
            x2, y2 = self._map_coords(x + 0.1, y + 0.1)
            self.w.create_oval(x1, y1, x2, y2, fill = "red")
            
        # Update text
        self.w.delete(self.text)
        self.time += 1
        self.text = self.w.create_text(
            25, 0, anchor=tk.NW,
            text=self._status_string(self.time, tank.getNumVisitedTiles()))
        self.master.update()
        time.sleep(self.delay)

    def done(self):
        "Indicate that the animation is done so that we allow the user to close the window."
        tk.mainloop()

