import numpy as np
import moviepy.video.io.VideoFileClip as mpy
import cv2
import collections
import moviepy.editor as mpy_
import datetime
from tqdm import tqdm

def take_derivative(I):
    x,y = I.shape
    der = np.zeros((x,y,2))
    der[1:-1,:,0] = (I[2:,:] - I[:-2,:])/2
    der[:,1:-1,1] = (I[:,2:] - I[:,:-2])/2

    return der

def lucas_kanade(frame_before,frame_after,number=1):
    frame_before = frame_before / 255.
    frame_after = frame_after / 255.

    dI = take_derivative(frame_after)
    Ix = dI[:,:,0].flatten()
    Iy = dI[:,:,1].flatten()

    It = (frame_after-frame_before).flatten()/number
    A = np.array([Ix,Iy]).T

    ATA = A.T@A
    ATA_inv = np.linalg.inv(ATA)
    uv = ATA_inv@(A.T@It)

    return uv

def find_hand(frame):
    frame[frame<191] = 0
    hands = np.ma.masked_equal(frame,0)

    coor_hands = np.nonzero(hands)
    coor_hands[0][coor_hands[0]>400] = 0
    coor_hands[1][coor_hands[0]==0] = 0
    right_hand = np.ma.masked_equal(coor_hands,0)

    return right_hand

def get_center(frame1,frame2):
    right_hand1 = find_hand(frame1)
    right_hand2 = find_hand(frame2)

    m_r = max(np.max(right_hand1[0]),np.max(right_hand2[0]))
    m_l = min(np.min(right_hand1[0]),np.min(right_hand2[0]))
    m_d = max(np.max(right_hand1[1]),np.max(right_hand2[1]))
    m_u = min(np.min(right_hand1[1]),np.min(right_hand2[1]))

    center_x = int((m_r+m_l)/2)
    center_y = int((m_d+m_u)/2)
    width_half = int(max(m_r-m_l,m_d-m_u)/2)

    return (center_x,center_y),width_half

if __name__ == "__main__":
    biped_vid = mpy.VideoFileClip("biped_1.avi")

    length = 5
    count = 0.

    frame_count = biped_vid.reader.nframes
    video_fps = biped_vid.fps

    frames = collections.deque([], length)    
    frames_color = collections.deque([], int(length/2))
    result = []

    old_time = datetime.datetime.now()
    # tqdm(range(frame_count), unit =" sample", desc ="Low Passing... "):
    fps = "0"
    for i in tqdm(range(frame_count), unit =" sample", desc ="Processing... "):
        walker_frame = biped_vid.get_frame(i*1.0/video_fps)
        frames_color.appendleft(walker_frame)
        wf_gray = cv2.cvtColor(walker_frame, cv2.COLOR_BGR2GRAY)

        frames.appendleft(wf_gray)
        last_uv = np.array([0.,0.])
        count = 0.

        if i>4:
            for j in range(len(frames)):
                if j!=2:
                    center,width_half = get_center(frames[2],frames[j])

                    hand_1 = frames[2][center[0]-width_half : center[0]+width_half, center[1]-width_half : center[1]+width_half]
                    hand_2 = frames[j][center[0]-width_half : center[0]+width_half, center[1]-width_half : center[1]+width_half]
                    # cv2.imshow("hand_1",hand_1)
                    # cv2.imshow("hand_2",hand_2)
                    # cv2.waitKey(0)
                    if j<2:
                        uv = lucas_kanade(frame_before=hand_2,frame_after=hand_1,number=abs(2-j))
                    else:
                        uv = lucas_kanade(frame_before=hand_1,frame_after=hand_2,number=abs(2-j))

                    last_uv += uv*(1/abs(2-j))
                    count += 1/abs(2-j)

            last_uv = last_uv/count
            arrowed = cv2.arrowedLine(frames_color[-1], pt1=(center[1],center[0]), pt2=(center[1]+int(last_uv[1]*100), center[0]+int(last_uv[0]*100)), color=(255, 0, 0), thickness=2)
            result.append(arrowed)
        time = datetime.datetime.now()
        # print("FPS: ",1/(time-old_time).total_seconds())
        fps = str(1/(time-old_time).total_seconds())
        old_time = time

    clip = mpy_.ImageSequenceClip(result, fps=5)
    clip.write_videofile('LucasKanade.mp4', codec='libx264')


