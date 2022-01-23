import cv2
import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('video_file', type=str)
args = parser.parse_args()


vidcap = cv2.VideoCapture("input_videos/"+args.video_file)
fps = vidcap.get(cv2.CAP_PROP_FPS)
vidname = args.video_file.split('.')[0]
success,image = vidcap.read()
count = 0
while success:
    if not os.path.isdir("input_videos/"+vidname):
        os.mkdir("input_videos/"+vidname)
    cv2.imwrite("input_videos/{}/frame%d.jpg".format(vidname) % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    if not success:
        print('Read a new frame: ', count, True)
    else:
        print('Read a new frame: ', count, success)
    count += 1
print('fps:',fps)

with open('batch.bat', 'w') as f:
    for i in range(count): 
        f.write('realesrgan-ncnn-vulkan.exe -i input_videos/{}/frame{}.jpg -o frame{}.png\n'.format(vidname, i, i))
    f.write('ffmpeg -i frame%%d.png -c:v libx264 -vf fps={} -pix_fmt yuv420p {}_result.mp4\n'.format(fps, vidname))
    f.write('mkdir {}\n'.format(vidname))
    for i in range(count):
        f.write('move frame{}.png {}\n'.format(i, vidname))

    # for i in range(count):
    #     f.write('del /f frame{}.png\n'.format(i))

p = subprocess.Popen("batch.bat")
stdout, stderr = p.communicate()
