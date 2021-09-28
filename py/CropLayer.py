class CropLayer(object):
    def __init__(self, params, blobs):
        # X,Y coordinates for the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, input):
        # Crops first input to match second input, keeping batch size and number of channels
        (inputShape, targetShape) = (input[0], input[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # Compute starting and end crop coords
        self.startX = int((inputShape[3] - targetShape[3])/2)
        self.startY = int((inputShape[2] - targetShape[2])/2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]
    
    def forward(self, inputs):
        # Use derived X,Y coordinates to crop
        return [inputs[0][:,:,self.startY:self.endY, self.startX:self.endX]]
