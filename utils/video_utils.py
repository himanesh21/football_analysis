import cv2

def read_video(video_path):
    # Read the video from the specified path
    cap = cv2.VideoCapture(video_path)
    frames=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
    
def save_video(output_frames,output_path):
    # Save the video to the specified path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (output_frames[0].shape[1],output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release

def visual_frames(output_frames):
    # Display the frames
    for frame in output_frames:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        