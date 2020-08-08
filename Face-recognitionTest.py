import face_recognition

imageYang = face_recognition.load_image_file("image_character/yang1.jpg")
face_locations = face_recognition.face_locations(imageYang)