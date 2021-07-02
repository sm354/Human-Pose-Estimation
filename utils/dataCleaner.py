import os

count = 0
for f in os.listdir("./Cropped_2/"):
    name = f[:-4]
    if len(name) < 3:
        print(f)
        count += 1
    else:
        os.remove("./Cropped_2/"+f)