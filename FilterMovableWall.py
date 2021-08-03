import numpy as np
import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpy_
import cv2
from LucasKanade import take_derivative, lucas_kanade
import collections
import datetime
from FilterWall import blurring, hand_getter, get_hand, get_center
from tqdm import tqdm

def calc_error(uv1,uv2):
    sqr = np.power((uv1-uv2),2)
    sqr_ = sqr[0] + sqr[1]
    root = np.power(sqr_,0.5)

    return root,sqr


if __name__ == "__main__":
    biped_vid = mpy.VideoFileClip("biped_3.avi")

    frame_count = biped_vid.reader.nframes
    video_fps = biped_vid.fps

    wall_const = 3000
    rog_const = 3000
    length=5
    const_wall = -1.3
    region_xs = np.array([205,295])
    region_ys = np.array([242,306])
    center_wall = (int((205+295)/2),int((242+306)/2))

    frames_wall = collections.deque([], length) 
    frames_color_wall = collections.deque([], int(length))

    count = 0.
    error = 0.
    error_arr = np.zeros((2))

    with open('sayac.npy', 'rb') as f:
        truth = np.load(f)

    old_time = datetime.datetime.now()    
    middle = int(length/2)
    result = []

    for i in tqdm(range(frame_count)):
        walker_frame = biped_vid.get_frame(i*1.0/video_fps)
        frames_color_wall.appendleft(walker_frame)
        wf_gray = cv2.cvtColor(walker_frame, cv2.COLOR_BGR2GRAY)

        last_uv_wall = np.array([0.,0.])
        last_uv_rog = np.array([0.,0.])

        region_edges = blurring(wf_gray)[region_xs[0]:region_xs[1], region_ys[0]:region_ys[1]]
        frames_wall.appendleft(region_edges)

        if i>length-1:            
            for j in range(len(frames_wall)):
                if j!=middle:
                    center_rog,width_half = get_center(frames_color_wall[middle],frames_color_wall[j])
                    hand_1 = frames_color_wall[middle][center_rog[0]-width_half : center_rog[0]+width_half, center_rog[1]-width_half : center_rog[1]+width_half]
                    hand_2 = frames_color_wall[j][center_rog[0]-width_half : center_rog[0]+width_half, center_rog[1]-width_half : center_rog[1]+width_half]

                    hand_1 = hand_getter(np.copy(hand_1))
                    hand_2 = hand_getter(np.copy(hand_2))

                    if j<middle:
                        uv_wall = lucas_kanade(frame_before=frames_wall[j],frame_after=frames_wall[middle],number=abs(middle-j))
                        uv_rog = lucas_kanade(frame_before=hand_2,frame_after=hand_1,number=abs(middle-j))
                    elif j>middle:
                        uv_wall = lucas_kanade(frame_before=frames_wall[middle],frame_after=frames_wall[j],number=abs(middle-j))
                        uv_rog = lucas_kanade(frame_before=hand_1,frame_after=hand_2,number=abs(middle-j))

                    last_uv_wall += uv_wall*(1/abs(middle-j))
                    last_uv_rog += uv_rog*(1/abs(middle-j))
                    count += 1/abs(middle-j)

            last_uv_wall = last_uv_wall/count
            last_uv_rog = last_uv_rog/count
            last_uv_rog[0] -= const_wall*last_uv_wall[0]            # duvara göre hand OF unu düzeltme. Sadece düşey eksende yapılacağı için tek boyutlu toplama yapılmalıdır

            err,err_arr = calc_error(truth[i-length],last_uv_rog)
            error += err
            error_arr += err_arr

            time = datetime.datetime.now()
            # print("FPS: ",1/(time-old_time).total_seconds())
            old_time = time            

            arrowed_wall = cv2.arrowedLine(frames_color_wall[middle+1], pt1=(center_wall[1],center_wall[0]), pt2=(center_wall[1]+int(last_uv_wall[1]*wall_const), center_wall[0]+int(last_uv_wall[0]*wall_const)), color=(255, 0, 0), thickness=2)
            arrowed_rog = cv2.arrowedLine(arrowed_wall, pt1=(center_rog[1],center_rog[0]), pt2=(center_rog[1]+int(last_uv_rog[1]*rog_const), center_rog[0]+int(last_uv_rog[0]*rog_const)), color=(0, 0, 255), thickness=2)
            result.append(arrowed_rog)

    clip = mpy_.ImageSequenceClip(result, fps=5)
    clip.write_videofile('FilterMovableWall.mp4', codec='libx264')

    print("Error: ",error/truth.shape[0])
    print("error_arr: ",error_arr/truth.shape[0])