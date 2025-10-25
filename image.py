import cv2

# split image into halves
def split_img(filepath):
    image = cv2.imread(filepath)
    height, width = image.shape[:2]

    mid_x = width // 2

    left_half = image[:, :mid_x]
    right_half = image[:, mid_x:]

    cv2.imwrite(f"{filepath}_left.jpg", left_half)
    cv2.imwrite(f"{filepath}_right.jpg", right_half)

if __name__ == "__main__":
    split_img("2col.jpg_dewarped.jpg")