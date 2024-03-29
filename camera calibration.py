import numpy as np
import cv2
import glob

# 棋盘格设置
chessboard_size = (7,11)  # 棋盘格角点的数量 (宽度角点数, 高度角点数)
square_size = 20  # 棋盘格每个方格的大小，单位可以是毫米或者英寸

# 准备棋盘格角点的3D坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储所有图片的3D点和2D点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 读取图片
images = glob.glob('C:/Users/131A2AB/Documents/MVDCPImages/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到足够的角点，添加到点集
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"棋盘格在图片 {fname} 上未被检测到。")

cv2.destroyAllWindows()

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机内参矩阵:\n", camera_matrix)
print("畸变系数:\n", dist_coeffs)
