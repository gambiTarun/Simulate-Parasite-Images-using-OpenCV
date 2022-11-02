# simulate-parasite-images
Creating simulated high resolution images of random shaped parasites and their blood vessel network

# Introduction
This objective of this project is to create simulated data for microbiologist researchers who are investigating cancer in parasitic microorganisms. The focus of this project is to implement different image processing techniques to create these "simulated images" and find an optimal way to compress these high resolution images to store them efficiently.

The requirements are to create two images corresponding to one parasite. One is the sensor image, which is the image of the parasite itself consisting of binary pixel values where 1 corresponds to the background and 0 corresponds to the parasite body. The microscope has zoomed in on each parasite so that the
parasite occupies 25% or more of the total area of the image. The other is the dye image, which is an image captured by the dye sensor. The dye is a luminescent dye injected into the parasite colony and should highlight the blood vessels inside the parasite body. Unfortunately, there was a dye leak and hence there might be dye present outside the parasite body as well. The image from the dye sensor will again have binary pixel values where 1 corresponds to the backgrounf and 0 corresponds to presence of dye at the location. 

Then comes the task to come up with a data structure to compress these high resolution images.

We also have to write functions to detect cancer in the parasite using the two images of the parasite and the dye. A parasite is deemed to have cancer if the total amount of dye detected in its body exceeds 10% of the area occupied by the parasite in the image. It is expected that fewer than 0.1% of the parasites will have cancer. Furthermore, we will write another function with a more efficient approach to detect cancer.

### Assumption
I assume in this simulation that the parasite body is globular and has a convex shape.

# Creating a random globular shape
The objective is to come up with a random globular shape for every parasite image. The approach I use here is to take a bunch of random points on a 2D plane and create their convex hull. The convex hull will give the smallest convex polygon containing all the given points. Then I have to smoothen this random convex polygon to create a smooth globular shape and while making sure it is still a convex shape. To do this I remove the higher frequencies in the Fourier Transform of the edges of polygon. This results in a smooth interpolation between the vertices of the convex hull polygon.

The following image shows the random points, the convex hull poygon, and the smooth interpolated blob in a simulation

![creatingGlob](https://user-images.githubusercontent.com/22619455/199347864-516a7eca-e3e7-4b5c-a3d0-92e3547ef200.jpg)

# Creating the parasite image 
After getting the globular shape, now I can use these coordinates as boundary locations for the parasite in the final image. To convert this set of coordinates to pixel location on an image of desired resolution (width=1000, height=1000), I will take these glob coordinates, rescale them from 0-1 to blob range such that they fill atleast 25% of the area, and multiply it with the width and height of the final image. Converting the values to integer, I will get pixel coordinates corresponding to the boundary shape of the blob on my base image. There is an issue however, since the coordinates are multiplied and casted to integer the coordinate values will not be continuous and will be very sparse on the final image. The following image is made (note that currently the background is of value 0, and parasite has value 1)

![imagFromHull](https://user-images.githubusercontent.com/22619455/199352842-0d1b99a0-c19d-4bde-8bc4-23cd6b2c408d.jpg)

This is where I use various image morphing techniques from the openCV library to get the desired results.
I will use the "dilate" operation from openCV on the above binary image using an "elliptical kernel" of size 1% of the actual image. 

```
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width//100,height//100))
blobImag = cv2.dilate(blobImag, kernel, iterations=1)
```

![imagAfterDilation](https://user-images.githubusercontent.com/22619455/199353119-f986ada8-c0a9-4c53-a0ff-ec3c80abe324.jpg)

Then I use "floodfill" operation from openCV to fill in the parasite body

```
cv2.floodFill(blobImag, None, (width//2,height//2), 255)
```

![imagAfterFill](https://user-images.githubusercontent.com/22619455/199353444-0147d56c-8fdc-4e62-b338-5930ad258ac4.jpg)

Then finally I invert the image for the final sensor image 

```
blobImag = cv2.bitwise_not(blobImag)
```

This results in the following final parasite image

![imagAfterInvert](https://user-images.githubusercontent.com/22619455/199353487-4d398e3d-c540-42d3-b802-f0d453d8c4f0.jpg)

# Creating the dye image 
For the dye image I need to simulate the structure of a vascular network from the parasite. Again, this has to be random for every simulation. I can image a network of straight lines on which I can apply the image morphing techniques from openCV. 
So, for the base I will start with random straight lines using openCV on my base image 

```
for i in range(100):
    r1,r2 = np.random.randint(height,size=2)
    c1,c2 = np.random.randint(width,size=2)
    cv2.line(
              dyeImag,
              pt1 = (r1,c1), pt2 = (r2,c2),
              color = 255,
              thickness = 1
            )
```

![dyeBase](https://user-images.githubusercontent.com/22619455/199354696-547e5cf4-6f62-4731-9956-67d538bbd601.jpg)

To control the dye strenght, I will use the "close" operation from openCV on the above image. This operation is a "dialation" operation followed by an "erosion" operation. I will use a "cross kernel" for the above operation since it embodies a vascular network shape. For the dye strength, I will create a random variable with a desired probability distribution. This will ensure at max 0.1% of the cases will have cancer. I will give use this random variable for number of iterations in the "close" operation, since iteration of value 4 simulates very high dye strength and always results in more than 10% of the body region as dyed.

```
dyeStrength = np.random.choice([1,2,3,4], p=[0.399,0.6,0.0002,0.0008])

# applying a 'close' morphology operation to bold the vascular network 
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (width//220,height//220))
dyeImag = cv2.morphologyEx(dyeImag, cv2.MORPH_CLOSE, kernel, iterations=dyeStrength)
```

When Iteration=1

![dyeBoldIter1](https://user-images.githubusercontent.com/22619455/199362032-b0d99c61-c1dd-4c0f-b956-b384fba045c6.jpg)

When Iteration=3

![dyeBoldIter3](https://user-images.githubusercontent.com/22619455/199362035-89f6261c-3464-4c1d-9908-665b8efd6288.jpg)

To make this look more organic, I will apply the "erode" operation from openCV using the same kernel as above. And invert to get the final image

```
dyeImag = cv2.morphologyEx(dyeImag, cv2.MORPH_ERODE, kernel)
```

![dyeErode](https://user-images.githubusercontent.com/22619455/199362036-c9fb8530-eed2-41b6-ac96-aeedde65b9ac.jpg)

```
dyeImag = cv2.bitwise_not(dyeImag)
```

This results in the following final dye image

![dyeAfterInvert](https://user-images.githubusercontent.com/22619455/199362028-70dff1fe-e224-4f59-9839-d861f941ed61.jpg)

# Determine parasite having Cancer
For deciding if the parasite has cancer or not, I count the pixels in the sensor image corresponding to the parasite body. Also I count the pixel in the dye image (masked with sensor image, to get dye inside parasite) corresponding to dye. Then I check if the number of dyed pixels is more than 10% of the parasite body pixels. If yes, then the parasite has cancer

```
def hasCancer(sensImage, dyeImage):
    pixParasite, countParasite = np.unique(sensImage, return_counts=True)
    pixDyeInParasite, countDyeInParasite = np.unique(cv2.bitwise_or(sensImage,dyeImage), return_counts=True)

    return bool(100 * countDyeInParasite[pixDyeInParasite==0] / countParasite[pixParasite==0] > 10)
```

A more efficient way would be to count the pixels would be to just sum the locations with the desired pixel values. 

```
def hasCancerEfficient(sensImage, dyeImage):
    countParasite = np.sum(sensImage==0)
    countDyeInParasite = np.sum(cv2.bitwise_or(sensImage,dyeImage)==0)
    
    return bool(100 * countDyeInParasite / countParasite > 10)
```

# Sensor and Dye image

![finalImags](https://user-images.githubusercontent.com/22619455/199363909-8cd99254-c2dc-4fa8-8bd0-2a8298cbe027.jpg)

# Compressing the images

### The sensor image
Data Structure: A dictionary with key as row number and value of list with min and max column number for the range of the parasite blob in that row.

I will use the assumption here that the blob will always be of a convex shape. This will ensure that the range of columns will always be continuous as the body of the parasite. This can be controlled through the globularity parameter in the CreateParasiteImage(). In the worst case, it will be an image of a parasite that occupy all the rows in the image. For a case where image is of size 100000 x 100000, in the worst case we will require to store 1e5 row coordinates and 2*1e5 min/max column coordinates. With each number as uint32, this ammounts to 1.2 MB vs 2.5 GB (if we use 2 bits for each pixel) required to store a sensor image of the parasite. 

Reason: Since a convex shape cannot have two points inside the shape that connect through outside the shape, the starting and ending coordinates of the blob in each row will be enough to restore the original image. 

### The dye image
Data Structure: Since the dye image will be a random sparse image in terms of dyed pixels, we can store the dyed pixel coordinates in a row and col array. This will result in reduction of storage requirement for the image. The expected dyed region is around 10% of the body of the parasite. In the worst case let us assume it to be 10% of the entire image (as there will be some dye leaks). For a case where image is of size 100000 x 100000, in the worst case, it will require to store 1e10*0.1 = 1e9 coordinates. With each number as uint32, this ammounts to ~8 GB (row and column) of data vs 2.5 GB (if we use 2 bits for each pixel) required to store the original image. Therefore, we cannot compress the dye image.

Reason: A random blood vessel network does not have any pattern that can be used to store the image efficiently. 

# References
- [Morphological Transformations in OpenCV](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Convex Hull in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
- [FFT in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html)
