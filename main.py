import face_recognition
from PIL import Image, ImageDraw


def face_rec():
    gal_face_img = face_recognition.load_image_file('img/19a713c5edb6c69c5a0c31b875d732b3.jpg')
    gal_face_location = face_recognition.face_locations(gal_face_img)

    justice_league_img = face_recognition.load_image_file('img/1506288441_0_1106_5000_3937_600x0_80_0_0_1f54310449c422971d99a4f25eadeddd.jpg')
    justice_league_faces_locations = face_recognition.face_locations(justice_league_img)

    # print(gal_face_location)
    # print(justice_league_faces_locations)
    # print(f'Found {len(gal_face_location)} face(s) in this image')
    # print(f'Found {len(justice_league_faces_locations)} face(s) in this image')

    pil_img1 = Image.fromarray(gal_face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for(top, right, bottom, left) in gal_face_location:
        draw1.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

        del draw1
        pil_img1.save('img/new_gal1.jpg')

    pil_img2 = Image.fromarray(justice_league_img)
    draw2 = ImageDraw.Draw(pil_img2)

    for(top, right, bottom, left) in justice_league_faces_locations:
        draw2.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw2
    pil_img2.save("img/new_justice_league.jpg")


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(faces)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f'img/face{count}.jpg')
        count += 1

    return f'Found {count} faces'


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    print(img1_encodings)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]
    print(img2_encodings)

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)

    return result


def main():
    # face_rec()
    # print(extracting_faces('img/1506288441_0_1106_5000_3937_600x0_80_0_0_1f54310449c422971d99a4f25eadeddd.jpg'))
    print(compare_faces('img/face1.jpg', 'img/face5.jpg'))


if __name__ == '__main__':
    main()
