# 比较两个图片中人物面部相似度
import cv2
# import numpy as np
import face_recognition

imgCheng = face_recognition.load_image_file("image_character/jack cheng.jpg")
imgCheng = cv2.cvtColor(imgCheng, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("image_character/jack cheng test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceCheng = face_recognition.face_locations(imgCheng)[0]
# 面部数据
encodeCheng = face_recognition.face_encodings(imgCheng)[0]
# 绘制人物面部边框
cv2.rectangle(imgCheng, (faceCheng[3], faceCheng[0]), (faceCheng[1], faceCheng[2]), (255, 0, 255), 2)

faceTest = face_recognition.face_locations(imgTest)[0]
# 面部数据
encodeTest = face_recognition.face_encodings(imgTest)[0]
# 绘制人物面部边框
cv2.rectangle(imgTest, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (255, 0, 255), 2)

# 图片的长宽分别赋值给x，y
# x, y = imgCheng.shape[0:2]
# 图片长宽都缩放到原来的四分之一
# imgCheng = cv2.resize(imgCheng, (int(y / 4), int(x / 4)))

# 最近邻插值法缩放到原图的四分之一后显示
# imgCheng = cv2.resize(imgCheng, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

# 比较人物面部是否同一人，true: 同一人，false: 不是同一人
result = face_recognition.compare_faces([encodeCheng], encodeTest)
# 面部相似度（数值越低相似度越高）
faceDis = face_recognition.face_distance([encodeCheng], encodeTest)
# 图片上面写入面部比较结果和相似度数值
cv2.putText(imgTest, f"{result} {round(faceDis[0], 3)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
print(result, faceDis)

# 显示图片
cv2.imshow("jack cheng", imgCheng)
cv2.imshow("cheng test", imgTest)

cv2.waitKey(0)