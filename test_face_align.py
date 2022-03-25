import face_alignment_of as face_alignment
from skimage import io

s = 'face_alignment.detection.sfd'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)
print(preds)