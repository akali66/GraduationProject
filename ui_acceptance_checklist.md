# UI Acceptance Checklist

This checklist acts as a QA sign-off document for developers implementing the Drilling-Hole Circle-Detection App UI.

## 1. General App Layout

*   [ ] Main application window opens centered and spans a minimum of 1280x720 pixels.
*   [ ] Screen is divided into four distinct regions: Left Image Pane, Middle Control Pane, Right Results Pane, and Bottom Status Bar.
*   [ ] Window properly scales on 1080p high DPI Windows 11 monitors.
*   [ ] All fonts are neutral and easily legible. Grayscale/system-default styling is used.

## 2. Left Image Pane (Input/Output Display)

*   [ ] The image panel allows the display of an entire image, scaling to fit the panel bounds.
*   [ ] Bounding boxes or drawn circles explicitly update on top of the base image.
*   [ ] Panel contains a title/header identifying the region.

## 3. Middle Control Pane (Parameters & Actions)

*   [ ] Exists one "Load Image" trigger mechanism (button or menu).
*   [ ] Contains a dropdown titled "Detection Method" with at least: Hough Circle Transform, MinEnclosing, Canny+Hough, and Multi-scale Fusion.
*   [ ] Exists a "Enable Compare Mode" checkbox.
*   [ ] Includes `Canny Low Threshold` slider `[0-255]` with a default of 50.
*   [ ] Includes `Canny High Threshold` slider `[0-255]` with a default of 150.
*   [ ] Includes `Hough dp` slider `[1.0-3.0]` with default 1.2 and steps of 0.1.
*   [ ] Includes `Hough Min Distance` slider `[1-500]` with a default of 30.
*   [ ] Includes `Hough Param1` slider `[1-300]` with a default of 50.
*   [ ] Includes `Hough Param2` slider `[1-200]` with a default of 30.
*   [ ] Includes `Min Radius` slider `[1-200]` with a default of 10.
*   [ ] Includes `Max Radius` slider `[10-500]` with a default of 100.
*   [ ] "Run Detection" button is clearly visible.
*   [ ] "Export Results" button becomes active after a detection run or compare run.
*   [ ] **NOTE:** Sliders were explicitly chosen over numeric inputs for rapid parameter experimentation.

## 4. Right Results Pane (Metrics)

*   [ ] Text readout or tree-view area exists for displaying general metrics (e.g., number of circles found).
*   [ ] Under "Compare Mode", the tabular structure displays rows per algorithm and columns for Method, Circles Found, Time(s), and IoU (if ground truth available).

## 5. Bottom Status Pane

*   [ ] A persistent label runs across the bottom confirming user actions (e.g., "Ready", "Loading Image...", "Run successful. Found 42 holes.").

## 6. User Flows

*   [ ] **Single Method Run**: Load Image -> Select "Hough" -> Adjust Threshold Sliders -> Click "Run Detection" -> Overlaid Image Appears + Status says "Found X circles".
*   [ ] **Compare Mode Run**: Load Image -> Check "Enable Compare Mode" -> Click "Run Detection" -> Metrics block populates with data from all configured algorithms -> Click "Export Results" -> CSV or PNG dialog opens.
