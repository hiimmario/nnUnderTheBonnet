import glob

for index, image_file_name in enumerate(glob.glob("bottles_testset_sub_/*")):
    print(image_file_name)