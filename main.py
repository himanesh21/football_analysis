from utils import save_video,read_video,visual_frames,make_directories
from trackers import Tracker 
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from view_transformer import ViewTransformer

def main():
    """
    Orchestrates the complete visual analytics pipeline on a football video.

    Workflow:
    ---------
    1. Creates necessary folder structure.
    2. Loads input video frames.
    3. Tracks players and ball using BoT-SORT tracker.
    4. Adds player position coordinates per frame.
    5. Estimates and adjusts for camera movement frame-by-frame.
    6. Transforms adjusted positions to real-world coordinates.
    7. Interpolates missing ball positions.
    8. Computes speed and total distance per player.
    9. Draws:
        - Bounding boxes and labels.
        - Speed and distance annotations.
        - Camera movement information.
    10. Saves the final output video.

    Output:
    -------
    - Saves annotated video to `./outputs/tracked_fnl4.mp4`
    - Prints confirmation message.
    """


    folders=["caches","outputs","inputs"]
    make_directories(folders) # Ensure all necessary folders exist

    video_frames=read_video("./inputs/15sec_input_720p.mp4")
    
    # Initialize tracker and perform object tracking
    tracker = Tracker('./models/best.pt')
    tracks =tracker.object_tracker(tracker_name="botSORT", #Options: "botSORT", "byteTRACK"
                                   frames=video_frames
                                   ,confidence_threshold=0.7
                                   , read_from_cache=False,
                                   cache_path="./caches/player_track.pkl"
                                   )
    tracker.add_position_to_tracks(tracks)
    
    
    # Estimate and compensate for camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_cache=False,
                                                                                cache_path='caches/camera_movement_cache.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # Apply real-world coordinate transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Fill in missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
    # Compute player speed and total distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    # Draw annotations, movement, and speed info
    output_video_frames = tracker.draw_annotations(video_frames=video_frames,tracks=tracks)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save final video
    save_video(output_frames=output_video_frames,output_path="./outputs/tracked_fnl4.mp4")
    
    print("video tracked successfully")

if __name__=="__main__":
    main()