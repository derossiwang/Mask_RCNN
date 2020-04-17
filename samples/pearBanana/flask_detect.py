

# class_names = ['BG', 'banana', 'pear']
class_names = ['BG', 'pear', 'banana-ripe', 'banana-nonRipe']

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0] ### the length of this will be the count of items found
print(r['class_ids'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])