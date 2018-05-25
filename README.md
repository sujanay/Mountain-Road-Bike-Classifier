# Mountain-Road-Bike-Classifier
This work was accomplished using **jupyter notebook** in GCP. The images of mountain bike and road bike are
preprocessed using **keras** and the **CNN** model was implemented using **tensorflow**. 

## Training the model
- The training images were preprocessed using **keras** and fed to the CNN model.
- The input size of training images were 28x28.
- The model was trained using sophisticated AdamOptimizer with learning rate of 1e-4.

## Saving the model
- The trained model was saved using tensorflow **saver**.

## Testing the model
- During the test phase, the model was tested by retrieving the saved trained model against the test images.
- The test image along with the actual label, predicted label, and confidence score were displayed. 


