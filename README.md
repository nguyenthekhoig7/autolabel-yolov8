# autolabel-yolov8
A project that ultilize GroundedSAM model to annotate my dataset, using autodistill.

### Steps of the code:
- Start with a set of 7 zebra images
- Use autodistill-GroundedSAM to label those images
- Use the labeled image to train a YOLOv8 model
- See the performance of the trained YOLOv8 model on unseen image