import cv2
from simple_facerec import SimpleFacerec
import sqlite3 as sl
import time 
import argparse

def cctv(video_source=0):  # 0 for camera, or path to video file
    con = sl.connect('data.db')

    # Encode faces from a folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    print("Loaded known faces:", sfr.known_face_names)

    # Load Camera or Video
    cap = cv2.VideoCapture(video_source)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Keep original color frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for face recognition

        # Detect Faces
        face_locations, ids = sfr.detect_known_faces(frame_rgb)
        
        # Draw faces with different colors
        for face_loc, id in zip(face_locations, ids):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            name = ""
            if id == "Unknown":
                color = (0, 0, 255)  # Red for unknown (BGR format)
                name = "Unknown"
                text_color = (255, 255, 255)  # White text
            else:
                color = (0, 255, 0)  # Green for known
                query = "UPDATE USER SET location=?, time= ? where id=" + id
                data = ('Hostel O', time.time())
                with con:
                    con.execute(query, data)
                with con:
                    data = con.execute("SELECT name FROM USER WHERE id= " + id)
                    for row in data:
                        name = row[0]
                text_color = (0, 0, 0)  # Black text
                print(f"Recognized: {name} (ID: {id})")

            # Draw filled rectangle for name background
            cv2.rectangle(frame, (x1, y1-35), (x2, y1), color, cv2.FILLED)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add name text
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1
            cv2.putText(frame, name, (x1+6, y1-10), font, font_scale, text_color, font_thickness)

        # Display frame
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()

    if args.video:
        print(f"Processing video: {args.video}")
        cctv(args.video)
    else:
        print("Using camera feed")
        cctv(0)  # Use default camera



