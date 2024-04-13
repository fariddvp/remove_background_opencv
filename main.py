import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Global variables
selected_color = (0, 255, 0)  # Default color is green in RGBA format

def apply_color(event,color='green'):
    global selected_color
    selected_color = (0, 0, 0)

    if color == 'green':
        selected_color = (0, 255, 0)
    if color == 'blue':
        selected_color = (255, 0, 0)
    elif color == 'red':
        selected_color = (0, 0, 255)    
        
    # Update the background color
    background_color[:] = selected_color
    update_result()


def apply_back(event, back='image'):
    global image
    if back == 'image':
        img =  cv2.imread('/home/farid/Documents/TAAV_vscode_prj/remove_background_img/src/back.jpg')
        
        # Resize the image
        resized_image = cv2.resize(img, (image.shape[1], image.shape[0]))

        temp = cv2.bitwise_and(image, resized_image)

        # Combine foreground and background
        result = cv2.bitwise_or(foreground, temp)


        # Update the displayed image
        result_plot.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.draw()



def update_result():
    global image, mask, background_color

    background = cv2.bitwise_and(image, background_color)    

    # Resize the background mask to match the size of the foreground image
    resized_background_mask = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Combine foreground and background
    result = cv2.bitwise_or(foreground, resized_background_mask)

    # Update the displayed image
    result_plot.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.draw()

# Load the image
image = cv2.imread('/home/farid/Documents/TAAV_vscode_prj/remove_background_img/src/dog.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create mask
mask = np.zeros_like(gray)

# Draw contours on mask
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Apply mask to image
foreground = cv2.bitwise_and(image, image, mask=mask)

# Create blue background
background_color = np.zeros_like(image, dtype=np.uint8)
background_color[:] = (0, 255, 0)  # Default color is green in RGBA format

# Apply mask to default background
background_with_foreground_masked = cv2.bitwise_and(background_color, background_color, mask=cv2.bitwise_not(mask))

# Initialize result
result = np.copy(foreground)

# Display original and result images side by side
fig = plt.figure(figsize=(10, 5), facecolor='gray')

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Result image
plt.subplot(1, 2, 2)
result_plot = plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Result Image')
plt.axis('off')

# Create a color palette button
axcolor = fig.add_axes([0.1, 0.05, 0.15, 0.05])
palette_button = Button(axcolor, 'GREEN', color=(0,1,0), hovercolor='lightgray')

# Create a color palette button
axcolor1 = fig.add_axes([0.3, 0.05, 0.15, 0.05])
palette_button1 = Button(axcolor1, 'BLUE', color=(0,0,1), hovercolor='lightgray')

# Create a color palette button
axcolor2 = fig.add_axes([0.5, 0.05, 0.15, 0.05])
palette_button2 = Button(axcolor2, 'RED', color=(1,0, 0), hovercolor='lightgray')

# Create a color palette button
axcolor3 = fig.add_axes([0.7, 0.05, 0.15, 0.05])
palette_button3 = Button(axcolor3, 'IMAGE', color=(0,1,1), hovercolor='lightgray')



# Connect the color palette button to the function apply_color
palette_button.on_clicked(lambda event: apply_color(event, color='green'))
palette_button1.on_clicked(lambda event: apply_color(event, color='blue'))
palette_button2.on_clicked(lambda event: apply_color(event, color='red'))
palette_button3.on_clicked(lambda event: apply_back(event, back='image'))



plt.show()