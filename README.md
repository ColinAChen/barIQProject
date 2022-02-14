# JIA 1356 barIQProject
We are working with BarIQ, founded by Daniel Knotts, and a large distributor of Anheuser-Busch products to optimize inventory visibility and warehouse operations across the supply chain with the use of cameras. By utilizing machine learning, the program will learn to detect and identify the inventory, allowing warehouse managers to carry out tasks with ease. We will place an array of cameras that will take images of pallets at 3 different heights and use computer vision software to decipher the images, eliminating significant labor costs and time sinks. 


License: GNU AGPLv3
## Release Notes
### Version 0.1.0
#### New Features
- Developed the layout for a frontend multi-tab react app which serves as an inventory dashboard for warehouses and contains two tabs, one for displaying inventory data and one for uploading warehouse images 
- Created the entire first tab which serves as the actual dashboard, aside from a graph as well as any actual data that needs to get put in
#### Bug Fixes
- N/A
#### Issues
- An issue exists on the first tab with resizing of some of the div boxes that contain data on certain inventory analytics


## Other Project Details
Our project uses computer vision to identify beverages in pallets. Each pallet will have a flag tag hanging into the aisle. A tower with 2 cameras on each side will roll in front of each aisle taking scanning the products in the rows. Each side of the tower will have one bottom ultra-wide camera scanning the lowest row and the middle row and one top camera aimed at the top flags. The first tag in each row will have a barcode providing all the information needed by the client to determine exactly what product is there and other data on the pallets in that row. All the tags behind and including the first tag will have a neon green color allowing our software to easily identify that there is a tag. Our software will then count the pallets and update the inventory. Each row will have a left side of pallets and a right side, our software will crop the image into 2 images for each side and run our program on these 2 images. 
