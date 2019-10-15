#!/usr/local/bin/python3
from PIL import Image
import os
import random
# Open Paddington
if not os.path.exists('dataset/training_set/processed'):
    os.mkdir('dataset/training_set/processed')
for root, dirs, files in os.walk("dataset/training_set/cats/"):
	for file in files:
		if ".jpg" not in file:
			continue

		print("Processing: " + str(root) + str(file))
		img = Image.open(root + file)

# Resize smoothly down to 16x16 pixels
		imgSmall = img.resize((random.randint(16,44), random.randint(16,44)),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
		result = imgSmall.resize(img.size,Image.NEAREST)

# Save
		result.save('dataset/training_set/processed/' + file + '.jpg')
