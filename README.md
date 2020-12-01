# Augmenting Physical Textbooks
A CLI application that utilises a webcam to observe the user reading a book. When the user points to a particular section of the page, this indicates to the app that they would like more information. If more information can be offered to the user, the application will fetch the data and display it in the web browser. 

# Requirements
A webcam is required for the app to process a feed. 

NumPy is required for the app to run: https://numpy.org/ 

The app uses OpenCV version 3.3.1.11, which provides both video support and SIFT. You will need to pull down both the opencv and opencv_contrib repositories from GitHub and then compile and install OpenCV 3 from source: https://github.com/opencv/opencv, https://github.com/opencv/opencv_contrib

# Usage
Run the application from the command line using NumPy. 

Point to different areas on the page of the book, and the application should open more information in the web browser, if available. 

# Known Issues
If the result does not show, remove your finger from the frame and try again.

It can take the application a few seconds to recognise a fingertip on screen, you may need to point for a couple of seconds. 

# Release Notes
1.0.0
Initial release of the application.
