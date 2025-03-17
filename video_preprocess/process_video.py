# Copyright for parts this script by Zakharov et Al under CC BY-NC 4.0
# MonoHair https://doi.org/10.48550/arXiv.2403.18356
# Modified by Stretox
# Feel free to use ❤️

import cv2
import os
import math
import numpy as np
import argparse

def ReadVideo( video_path: str, save_root:str, num_frames: int, rotate:bool=True):
    print("Path to save to: ", save_root)
    if not os.path.exists(save_root):
        print("Creating Directory: ", save_root)
        os.makedirs(save_root)
    cap = cv2.VideoCapture(video_path)

    # Calculating how large the interval should be so we get num_frames=<frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = math.floor(total_frames/num_frames)

    i = 0
    frame_count = 0  # To keep track of the total number of frames processed

    while cap.isOpened():
        frames = []  # Reset frames for each batch
        for _ in range(interval):
            ret, frame = cap.read()
            if not ret:
                break  # Exit if there are no more frames to read
            frames.append(frame)
            frame_count += 1

        if len(frames) == 0:
            break  # Exit if no frames were read

        max_sharpless = 0
        max_frame = None
        max_i = -1

        for i, frame in enumerate(frames):
            # Rotate the frame if needed
            if rotate:
                (h, w) = frame.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((w / 2, h / 2), -90, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
                frame = cv2.warpAffine(frame, M, (nW, nH))

            # Calculate sharpness
            img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
            if imageVar > max_sharpless:
                max_frame = frame
                max_sharpless = imageVar
                max_i = i

        # Save the sharpest frame for the current batch
        if max_frame is not None:
            cv2.imwrite(os.path.join(save_root, '{}.png'.format(frame_count - interval + max_i)), max_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    cap.release()

    print("Finished extracting Frames! \n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sharpest frames from a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--num_frames', type=int, default=300, help='Number of frames to extract (default: 300)')
    parser.add_argument('--rotate', action='store_true', help='Rotate frames by 90 degrees counterclockwise')

    args = parser.parse_args()

    file = os.path.splitext(os.path.basename(args.video_path))[0]
    root = os.getcwd()
    save_root = os.path.join(root, f'{file}_output/')

    ReadVideo(args.video_path, save_root, args.num_frames, args.rotate)
