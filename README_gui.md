# Drilling-Hole Circle-Detection App GUI

A full-featured Tkinter desktop application for comparing multiple circle-detection algorithms (Hough, MinEnclosing, Canny+Hough, Multi-scale Fusion).

## How to Run

Activate your virtual environment and run the GUI application:
```cmd
python gui.py
```

## Features and Controls

*   **Load Image:** Click the button in the left pane to import a `.png`, `.jpg`, or `.bmp` file. The original image will display in the left frame.
*   **Detection Method Dropdown:** Select from one of the four implemented contour or Hough-based algorithms.
*   **Dynamic Parameter Sliders:** Modifying the dropdown updates the list of relevant tunable parameters instantly. Sliders automatically feed their values to the backend algorithms.
*   **Run Detection:** 
    *   Converts the image to grayscale and applies your selected detector.
    *   While processing, the cursor changes to a "busy" watch icon.
    *   *Success:* Draws a red circle (boundary) and green cross (center). The status bar updates with calculation time, radius, simulated coverage, and an abstract confidence score.
    *   *Failure:* Triggers a system warning or displays the failure directly.
*   **Compare Mode:** If checked, running detection will split the results canvas into four tiles and run **all** known methods concurrently, comparing them visually side-by-side using your current global parameters.
*   **Export Results:** Saves the final drawn visual element(s) as a high-resolution `.png` file. (In Compare mode, it exports a 2x2 collage).
*   **Save/Load Config:** Export the active parameters via the UI to a standalone JSON file for persistence, and load them back later.
