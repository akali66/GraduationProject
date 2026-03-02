# Algorithm Parameter Specification

## 1. Hough Circle Transform (Standard)

| Parameter Name          | Type  | Default | Allowed Range | Description                                      | Tuning Hints                                                                 |
| ----------------------- | ----- | ------- | ------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `dp`                    | Float | `1.2`   | `[1.0, 3.0]`  | Inverse ratio of resolution.                     | Higher values are faster but less accurate. Keep between 1.0 and 1.5.        |
| `minDist`               | Int   | `30`    | `[1, 500]`    | Minimum distance between centers.                | Increase if overlapping concentric circles are detected.                     |
| `param1` (Gradient)     | Int   | `50`    | `[1, 300]`    | Upper threshold for internal Canny edge detector.| Higher values detect stronger edges only.                                    |
| `param2` (Accumulator)  | Int   | `30`    | `[1, 200]`    | Accumulator threshold for circle centers.        | Lower values yield more false positive circles; higher values restrict them. |
| `minRadius`             | Int   | `10`    | `[1, 200]`    | Minimum radius to detect.                        | Set base size slightly below the expected minimum hole.                      |
| `maxRadius`             | Int   | `100`   | `[10, 500]`   | Maximum radius to detect.                        | Set upper bound tightly to avoid finding massive curves.                     |

## 2. MinEnclosing Circle (Contours)

| Parameter Name          | Type  | Default | Allowed Range | Description                                      | Tuning Hints                                                                 |
| ----------------------- | ----- | ------- | ------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `Binary Threshold`      | Int   | `128`   | `[0, 255]`    | Threshold for binarization prior to contours.    | Use Otsu's thresholding dynamically if lighting is inconsistent.              |
| `Min Contour Area`      | Int   | `100`   | `[0, 10000]`  | Filter out noise.                                | Determines how small a hole can be. Based on `minRadius` squared approx.     |

## 3. Canny + Hough Pipeline

| Parameter Name          | Type  | Default | Allowed Range | Description                                      | Tuning Hints                                                                 |
| ----------------------- | ----- | ------- | ------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `Canny Low Thresh`      | Int   | `50`    | `[0, 255]`    | Lower edge linking threshold.                    | Defines what constitutes a weak edge linking to a strong edge.               |
| `Canny High Thresh`     | Int   | `150`   | `[0, 255]`    | Upper edge threshold.                            | Defines the initial strong edges.                                            |
| *Inherits Hough Params* |       |         |               | Uses standard Hough parameters for post-Canny.   | See Section 1. `param1` is largely overridden by manual Canny step.          |

## 4. Multi-Scale Fusion

*This algorithm iteratively wraps the standard Hough over a pyramid of scales.*

| Parameter Name          | Type  | Default | Allowed Range | Description                                      | Tuning Hints                                                                 |
| ----------------------- | ----- | ------- | ------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `Scale Levels`          | Int   | `3`     | `[1, 5]`      | Number of pyramid levels to test.                | More levels take longer but handle widely varying drill hole sizes better.   |
| `Scale Factor`          | Float | `0.75`  | `[0.5, 0.9]`  | Image downscale factor per level.                | Standard is 0.75 or 0.5.                                                     |
| `NMS IoU Threshold`     | Float | `0.5`   | `[0.1, 0.9]`  | Non-Maximum Suppression overlap threshold.       | Lower values suppress more aggressively; use 0.5 to keep distinct adjacent holes. |
