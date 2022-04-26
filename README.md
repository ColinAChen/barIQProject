# JIA 1356 barIQProject
We are working with BarIQ, founded by Daniel Knotts, and a large distributor of Anheuser-Busch products to optimize inventory visibility and warehouse operations across the supply chain with the use of cameras. By utilizing machine learning, the program will learn to detect and identify the inventory, allowing warehouse managers to carry out tasks with ease. We will place an array of cameras that will take images of pallets at 3 different heights and use computer vision software to decipher the images, eliminating significant labor costs and time sinks. 

## Installation Guide
### Pre-Requisites/Troubleshooting
The primary pre-requisite for this application is to have Python 3+ installed. If encountering issues, first try updating Python by entering the following lines (make sure that you have first installed Homebrew, which is the first line in the Download Guide below):

  <code>brew install pyenv</code>
  
  <code>pyenv install 3.9.2</code>
  
If another error comes up, the first step is always to ensure that the correct libraries are installed. Check the error message and if a library seems missing, simply install using Homebrew:

  <code>brew install [insert library name]</code>

### Download Guide
First, install Homebrew, which will allow for an easy installation of the other necessary libraries:

  <code>/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"</code>
  
Next, install the following Python libraries using Pip, which comes with Homebrew:
  
  <code>pip install opencv-python</code>
  
  <code>pip install pyzbar</code>
  
  <code>pip install Flask</code>
  
Then, use npx, which comes with Homebrew, to install React and create the app framework:
  
  <code>npx create-react-app my-app</code>
 
### Starting the Flask server

In order to start the Flask server, run the following line in your terminal:

  <code>python app.py</code>
  
### Starting the React frontend

Finally, to start the front-end, run the following line:

  <code>npm start</code>
  
  

License: GNU AGPLv3
## Release Notes
### Version 1.0.0 (Latest)
#### New Features
- UI now includes a larger table for client to input and display more data
- Top 3 boxes on dashboard next to total inventory now show largest 3 brands by inventory count
#### Bug Fixes
- Algorithm for tag counting has been improved to minimize oversaturation of tags due to reflections

### Version 0.5.0
#### New Features
- UI now includes total counts of pellets separated by brand
- Total pellet counts across runs are visible on dashboard under pellet count boxes
#### Bug Fixes
- Front end image input is now fully functional in all cases without the need for additional requests
#### Issues
- Oversaturation of tags due to reflection occasionally results in loss of tag data

### Version 0.4.0
#### New Features
- Combined QR code and color recognition features into an integrated Python detector
- Tweaked some of the YOLO model's parameters to account for edge cases seen in warehouse simulation images
#### Bug Fixes
- Addressed color distortion due to shading and lighting by adding color flexibility parameter to recognition software
#### Issues
- Front end image input requests rarely do not register unless an additional request is queried

### Version 0.3.0
#### New Features
- Developed QR code recognition software in Python using various libraries/APIs
- Initialized Firebase database in JavaScript and Python to send data back and forth and also store pellet data
#### Bug Fixes
- Improved the integration between front-end and back-end by changing certain method calls and initialization parameters
#### Issues
- The QR code software occasionally fails to scan codes which occur on tags with very different lighting from the close tags

### Version 0.2.0
#### New Features
- Developed the flag-counting computer vision software in Python using OpenCV, a computer vision library 
- Integrated front-end image-inputting into React and Flask, another Python library, to allow for image processing
#### Bug Fixes
- Fixed the tab resizing issue in the div boxes from last release
#### Issues
- The image data being inputted from the front-end side is not formatting correctly in the algorithm pipeline

### Version 0.1.0
#### New Features
- Developed the layout for a frontend multi-tab react app which serves as an inventory dashboard for warehouses and contains two tabs, one for displaying inventory data and one for uploading warehouse images 
- Created the entire first tab which serves as the actual dashboard, aside from a graph as well as any actual data that needs to get put in
#### Issues
- An issue exists on the first tab with resizing of some of the div boxes that contain data on certain inventory analytics

## Other Project Details
Our project uses computer vision to identify beverages in pallets. Each pallet will have a flag tag hanging into the aisle. A tower with 2 cameras on each side will roll in front of each aisle taking scanning the products in the rows. Each side of the tower will have one bottom ultra-wide camera scanning the lowest row and the middle row and one top camera aimed at the top flags. The first tag in each row will have a barcode providing all the information needed by the client to determine exactly what product is there and other data on the pallets in that row. All the tags behind and including the first tag will have a neon green color allowing our software to easily identify that there is a tag. Our software will then count the pallets and update the inventory. Each row will have a left side of pallets and a right side, our software will crop the image into 2 images for each side and run our program on these 2 images.
