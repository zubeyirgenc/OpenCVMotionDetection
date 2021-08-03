import numpy as np
import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpy_
import cv2
from LucasKanade import take_derivative, lucas_kanade
import collections
import datetime
from tqdm import tqdm

def blurring(frame):                                                    # duvardan OF çıkarılabilir bir veri elde etmek için blurlama ve edge detection işlemi
    blur = cv2.GaussianBlur(frame,(11,11),0)
    canny = cv2.Canny(blur, 50, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.dilate(canny, kernel)
    return result

def hand_getter(frame):                                 #görseldeki eli threshold lar vasıtasıyla tespit etmek için
    green = frame[:,:,1]
    green[frame[:,:,0]>200] = 0
    green[frame[:,:,2]>200] = 0
    green[green<200] = 0
    return green

def get_hand(walker_frame):                             #tespit edilen elin konumunu almak işlemi
    green = hand_getter(walker_frame)
    green = cv2.GaussianBlur(green,(111,111),0)
    green[green<10] = 0
    green[green>10] = 255

    hands = np.ma.masked_equal(green,0)

    coor_hands = np.nonzero(hands)
    coor_hands[0][coor_hands[0]>400] = 0
    coor_hands[1][coor_hands[0]==0] = 0
    right_hand = np.ma.masked_equal(coor_hands,0)

    return right_hand

def get_center(frame1,frame2):                                  #bir önceki frame deki el ile şuanki eli kapsayacak kadar geniş en dar alanı tespit etmek için
    right_hand1 = get_hand(np.copy(frame1))
    right_hand2 = get_hand(np.copy(frame2))

    m_r = max(np.max(right_hand1[0]),np.max(right_hand2[0]))
    m_l = min(np.min(right_hand1[0]),np.min(right_hand2[0]))
    m_d = max(np.max(right_hand1[1]),np.max(right_hand2[1]))
    m_u = min(np.min(right_hand1[1]),np.min(right_hand2[1]))

    center_x = int((m_r+m_l)/2)
    center_y = int((m_d+m_u)/2)
    width_half = int(max(m_r-m_l,m_d-m_u)/2)

    return (center_x,center_y),width_half

if __name__ == "__main__":
    biped_vid = mpy.VideoFileClip("biped_2.avi")

    frame_count = biped_vid.reader.nframes
    video_fps = biped_vid.fps

    wall_const = 3000       #duvar uv sinin OF a dönüştürülürkenki katsayısı
    rog_const = 3000       #rogdoll uv sinin OF a dönüştürülürkenki katsayısı
    length=5                #kaç frane üzerinden bir OF çıkarılacağının sayısı. 5 için, n+2 n+1 n n-1 n-2
    region_xs = np.array([205,295])     #duvardan OF un çıkarılacağı alan
    region_ys = np.array([242,306])     #duvardan OF un çıkarılacağı alan
    center_wall = (int((205+295)/2),int((242+306)/2))   #OF tespiti için duvardaki alanın merkezi

    frames_wall = collections.deque([], length)         #son length kadarki frami tutulma queue si
    frames_color_wall = collections.deque([], int(length))           #son length kadarki renkli frame i tutulma queue si
    # frames_rog = collections.deque([], length) 

    
    count = 0.
    sayac = []
    result = []

    old_time = datetime.datetime.now()

    for i in tqdm(range(frame_count)):
        walker_frame = biped_vid.get_frame(i*1.0/video_fps)
        frames_color_wall.appendleft(walker_frame)
        wf_gray = cv2.cvtColor(walker_frame, cv2.COLOR_BGR2GRAY)
        # frames_rog.appendleft(wf_gray)
        last_uv_wall = np.array([0.,0.])        #length kadar ki UV lerin toplanıp ortalaması alınması için
        last_uv_rog = np.array([0.,0.])        #length kadar ki UV lerin toplanıp ortalaması alınması için

        region_edges = blurring(wf_gray)[region_xs[0]:region_xs[1], region_ys[0]:region_ys[1]]
        frames_wall.appendleft(region_edges)        #duvardaki seçilen bölegede edge detection yapılmış bölge

        if i>4:
            for j in range(len(frames_wall)):
                if j!=2:
                    center_rog,width_half = get_center(frames_color_wall[2],frames_color_wall[j])   #rogdoll un elinin iki frame deki merkezi ve kare şeklinde genişliği
                    hand_1 = frames_color_wall[2][center_rog[0]-width_half : center_rog[0]+width_half, center_rog[1]-width_half : center_rog[1]+width_half]     #tespit edilen koordinatlar üzerinden frame de bölge seçmek için
                    hand_2 = frames_color_wall[j][center_rog[0]-width_half : center_rog[0]+width_half, center_rog[1]-width_half : center_rog[1]+width_half]

                    hand_1 = hand_getter(np.copy(hand_1))   # koordinatlar üzerinden eli getirmek için
                    hand_2 = hand_getter(np.copy(hand_2))

                    if j<2:     #2 den büyük olanlarda önce gelen frame değiştiği için kontrol
                        uv_wall = lucas_kanade(frame_before=frames_wall[j],frame_after=frames_wall[2],number=abs(2-j))  #duvar OF u
                        uv_rog = lucas_kanade(frame_before=hand_2,frame_after=hand_1,number=abs(2-j))                   #rogdoll OF u
                    elif j>2:
                        uv_wall = lucas_kanade(frame_before=frames_wall[2],frame_after=frames_wall[j],number=abs(2-j))
                        uv_rog = lucas_kanade(frame_before=hand_1,frame_after=hand_2,number=abs(2-j))

                    last_uv_wall += uv_wall*(1/abs(2-j))
                    last_uv_rog += uv_rog*(1/abs(2-j))
                    count += 1/abs(2-j)

            last_uv_wall = last_uv_wall/count   #toplanan UV ların ortalamasını almak için
            last_uv_rog = last_uv_rog/count
            sayac.append(last_uv_rog)           #error hesabı için OF ları kaydetmek için

            time = datetime.datetime.now()
            # print("FPS: ",1/(time-old_time).total_seconds())
            old_time = time

            arrowed_wall = cv2.arrowedLine(frames_color_wall[3], pt1=(center_wall[1],center_wall[0]), pt2=(center_wall[1]+int(last_uv_wall[1]*wall_const), center_wall[0]+int(last_uv_wall[0]*wall_const)), color=(255, 0, 0), thickness=2)
            arrowed_rog = cv2.arrowedLine(arrowed_wall, pt1=(center_rog[1],center_rog[0]), pt2=(center_rog[1]+int(last_uv_rog[1]*rog_const), center_rog[0]+int(last_uv_rog[0]*rog_const)), color=(0, 0, 255), thickness=2)
            result.append(arrowed_rog)

    sayac = np.asarray(sayac)
    clip = mpy_.ImageSequenceClip(result, fps=5)
    clip.write_videofile('FilterWall.mp4', codec='libx264')

    with open('sayac.npy', 'wb') as f:
        np.save(f, sayac)           #error hesabı için OF ları kaydetmek için
