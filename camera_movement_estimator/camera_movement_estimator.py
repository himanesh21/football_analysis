import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    """
    A utility class to estimate and compensate for camera movement across video frames,
    useful in sports analytics, object tracking, or any scenario where camera panning/tilting
    needs to be accounted for.

    Functionality:
    -------------
    - Uses Lucas-Kanade optical flow to detect motion vectors of stable features.
    - Computes per-frame camera movement vectors based on dominant feature shifts.
    - Provides tools to adjust object positions to account for camera motion.
    - Supports optional result caching via pickle ("stub") files.
    - Allows visualization of estimated camera motion over the video.

    Attributes:
    -----------
    minimum_distance (float):
        Threshold distance to consider a movement as significant.
    
    lk_params (dict):
        Parameters for `cv2.calcOpticalFlowPyrLK` for dense optical flow tracking.
    
    features (dict):
        Parameters for `cv2.goodFeaturesToTrack`, limited to feature-rich zones 
        (e.g., edges of the frame assumed to include goalposts or sidelines).
    
    Example Usage:
    --------------
    >>> estimator = CameraMovementEstimator(initial_frame)
    >>> movement = estimator.get_camera_movement(frames)
    >>> estimator.add_adjust_positions_to_tracks(tracks, movement)
    >>> annotated_frames = estimator.draw_camera_movement(frames, movement)
    """
    def __init__(self,frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        """
    Adjusts tracked object positions based on estimated camera movement to provide 
    stabilized coordinates across frames.

    Args:
        tracks (dict): 
            Dictionary of tracked object positions across frames. 
            Format: {object_class: [{track_id: {'position': (x, y)}}]}.
        camera_movement_per_frame (List[List[float]]): 
            List of [x, y] camera movement vectors per frame.

    Modifies:
        Adds a new key `'position_adjusted'` to each tracked object containing its position 
        relative to a stabilized (static) camera.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if track is None:
                    continue
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self,frames,read_from_cache=False, cache_path=None):
        """
    Estimates the movement of the camera across a sequence of frames using optical flow.

    Args:
        frames (List[np.ndarray]): 
            List of video frames (BGR).
        read_from_cache (bool, optional): 
            If True, reads cached camera movement data from `cache_path` instead of recomputing.
        cache_path (str, optional): 
            Path to load/save the camera movement data via pickle.

    Returns:
        List[List[float]]: 
            A list containing [x_movement, y_movement] per frame. If movement is insignificant, 
            the movement is [0, 0].
    
    Notes:
        - Uses Lucas-Kanade optical flow with feature masking to restrict detection to goalposts.
        - Replaces good features only when a significant movement is detected.
        """
        # Read the cache 
        if read_from_cache and cache_path is not None and os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()
        
        if cache_path is not None:
            with open(cache_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        """
    Overlays the estimated camera movement vector on each frame for visualization.

    Args:
        frames (List[np.ndarray]): 
            List of original video frames.
        camera_movement_per_frame (List[List[float]]): 
            A list of [x, y] movement values corresponding to each frame.

    Returns:
        List[np.ndarray]: 
            Annotated video frames with camera movement information displayed in the top-left corner.

    Notes:
        - Draws a semi-transparent overlay for better text readability.
        - Displays camera movement in both X and Y directions.
        """
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames