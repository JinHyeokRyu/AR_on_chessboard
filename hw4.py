import cv2 as cv
import numpy as np

# settings
video_file = 'test.mp4'
K = np.array([[1.99289143e+03, 0, 5.22590339e+02],
              [0, 1.96989478e+03, 7.16385708e+02],
              [0, 0, 1]])
dist_coeff = np.array([-0.63361241, -0.41990132, 0.0048716, 0.0057259, 0.98918922])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK


# Prepare a 3D box for simple AR
pyramid = board_cellsize * np.array([[2, 1, 0], [6, 1, 0], [6, 5, 0], [2, 5, 0], [4, 3, -3]])
# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])


save_path = 'test_ar.mp4'



# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

fps = video.get(cv.CAP_PROP_FPS)
w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
writer = cv.VideoWriter(save_path, fourcc, fps, (w, h))


# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        pts, _ = cv.projectPoints(pyramid, rvec, tvec, K, dist_coeff)
        pts = np.int32(pts).reshape(-1,2)
        cv.polylines(img, [np.int32(pts[:4])], True, (0, 255, 0), 2)
        for i in range(4):
            cv.line(img, tuple(pts[i]), tuple(pts[4]), (0, 0, 255), 2)

    writer.write(img)

video.release()
writer.release()
print("Saved to:", save_path)