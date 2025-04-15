# start dlc
import deeplabcut as dlc

# check GPU (optional)
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# start projects
import glob
import os 
os.chdir('chage to your path')
videos = glob.glob('*.mp4')
learningVideo = ["add video name for learning"]

config_path = dlc.create_new_project('PJ name', 'my name', learningVideo, working_directory='add your directory', copy_videos=False)

# if the project was already created, run this
config_path = 'path to yaml file'


# extract frame
dlc.extract_frames(config_path, 'automatic', 'kmeans')

# label the data
run 'napari' on your Anaconda prompt

# check labels
#dlc.check_labels(config_path)

# creating training dataset
dlc.create_training_dataset(config_path, num_shuffles=1)

# start training
dlc.train_network(config_path)

# evaluate the trained network
#dlc.evaluate_network(config_path, plotting=True)

full_path_to_videos = []
root = 'add your path'
for path in videos:
    full_path_to_videos.append(root + '/' + path)

# video analysis and plotting results
dlc.analyze_videos(config_path, full_path_to_videos, shuffle=1, save_as_csv=False, videotype='.mp4')
dlc.filterpredictions(config_path, full_path_to_videos, shuffle=1, save_as_csv=True, videotype='.mp4')


videoCreate = ["add your videos"]
dlc.create_labeled_video(config_path, videoCreate, filtered=True)

# refine videos
refineVideos = ["add your videos"]
dlc.extract_outlier_frames(config_path, refineVideos, outlieralgorithm='jump')
dlc.refine_labels(config_path)
dlc.merge_datasets(config_path)
# then, back to "create_training_dataset()"
