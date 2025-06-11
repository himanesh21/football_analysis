from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import os
import supervision as sv
import cv2
from utils import get_bbox_width,get_center_of_bbox,get_foot_position
from boxmot import BoTSORT
import torch
from pathlib import Path
import numpy as np
import pandas as pd 


class Tracker:
    """
    Tracker class for detecting and tracking players, referees, goalkeepers, and the ball
    in football video frames using YOLO and ByteTrack/BOT-SORT.

    Attributes:
        model (YOLO): YOLO model for object detection.
        tracker (ByteTrack): Tracker for player and referee tracking.
        tracker2 (BoTSORT): BoT-SORT tracker with ReID for better tracking performance.
    """
    
    def __init__(self,model_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model =YOLO(model_path).eval().to(self.device)
        self.tracker = sv.ByteTrack()
        self.tracker2 = BoTSORT(
                model_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
    device=self.device,  # Use CPU for inference
    with_reid=True,fp16=False,
        )
    
    def add_position_to_tracks(sekf,tracks):

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():      
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolates missing ball bounding box positions across video frames.
    
        This method is useful in cases where the ball is not detected in certain frames.
        It performs linear interpolation followed by backfilling to estimate bounding boxes
        for those missing detections.

        Args:
            ball_positions (list of dict): 
                A list where each element is a dict of the form:
                `{1: {"bbox": [x1, y1, x2, y2]}}` representing the ball's bounding box 
                in a frame. If the ball is missing in a frame, the dict may be empty or contain no valid bbox.

        Returns:
            list of dict:
                A new list of the same length where each element contains the interpolated
                ball bounding box in the same `{1: {"bbox": [x1, y1, x2, y2]}}` format.
                Ensures that no frame is left with missing ball position
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    

    def detect_frames(self,frames):
        """
        Runs batched detection on input video frames using YOLO.

        Args:
            frames (list of ndarray): List of input video frames.

        Returns:
            list: Detection results per frame.
        """
        batch_size=20
        results=[]
        for i in range(0,len(frames),batch_size):
            detection_batch=self.model.predict(frames[i:i+batch_size],conf=0.7)
            results.extend(detection_batch)

        return results
        


    def get_object_track_with_byteTracker(self, frames, read_from_cache=False, cache_path=None):
        """
    Performs multi-object tracking using ByteTrack on a sequence of video frames.

    This method uses a YOLO model for object detection and the ByteTrack algorithm 
    for associating object detections across frames. It supports tracking of players, 
    referees, the ball, and goalkeepers.

    Args:
        frames (list of np.ndarray):
            List of video frames (as numpy arrays) to run detection and tracking on.
        
        read_from_cache (bool, optional):
            If True, attempts to load tracking results from the given `cache_path` instead 
            of computing from scratch. Default is False.
        
        cache_path (str, optional):
            Path to a pickle file where cached tracking results can be loaded from 
            or saved to. If None, caching is skipped.

    Returns:
        dict: A dictionary structured as:
            {
                "players":    [ {track_id: {"bbox": [x1, y1, x2, y2]}} for each frame ],
                "referees":   [ {...} ],
                "ball":       [ {...} ],
                "goalkeeper": [ {...} ]
            }
            Each list entry corresponds to one frame, and each object is identified 
            by a unique `track_id`.
    
    Notes:
        - The 'goalkeeper' class is re-labeled as 'player' before tracking, to unify 
          their treatment under the same tracking category.
        - The ball is not tracked using ByteTrack but instead is detected directly per frame.
        """
        if read_from_cache and cache_path is not None and os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[],
            "goalkeeper":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeeper"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}


        if cache_path is not None:
            with open(cache_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def format_detection(self,boxes,confidence_threshold): # Format detections: [x1, y1, x2, y2, conf, cls]
            dets = []
            for box in boxes:
                conf = box.conf[0].item()
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    dets.append([x1, y1, x2, y2, conf, cls_id])

            dets = np.array(dets)

            if dets.size == 0:
                dets = np.empty((0, 6))
            elif dets.ndim == 1:
                dets = dets.reshape(1, -1)
            elif dets.shape[1] > 6:
                dets = dets[:, :6]
            elif dets.shape[1] < 6:
                raise ValueError(f"Each detection must have 6 elements. Got shape {dets.shape}")
            return dets
        
    
    def get_object_track_with_botSORT(self,frames, confidence_threshold,read_from_cache=False, cache_path=None):
        """
         Performs multi-object tracking using BoT-SORT on a sequence of video frames.

         This method uses the YOLO detector to extract detections, filters them by confidence threshold,
         and applies the BoT-SORT tracking algorithm (with optional ReID) to assign consistent IDs 
         across frames. Supports tracking of players, referees, the ball, and goalkeepers.

         Args:
             frames (list of np.ndarray):
                 List of video frames (as numpy arrays) to perform detection and tracking on.

             confidence_threshold (float):
                 Minimum confidence score to consider a detection valid. Helps filter out low-confidence
                 bounding boxes before tracking.

             read_from_cache (bool, optional):
                 If True, attempts to load tracking results from the `cache_path` pickle file.
                 If the cache file exists, detection and tracking are skipped.

             cache_path (str, optional):
                 File path to load/save cached tracking results. If None, no caching is used.

         Returns:
             dict: A dictionary containing per-frame tracking results with this structure:
                 {
                     "players":    [ {track_id: {"bbox": [x1, y1, x2, y2]}} for each frame ],
                     "referees":   [ {...} ],
                     "ball":       [ {...} ],
                     "goalkeeper": [ {...} ]
                 }

         Notes:
             - The ball is treated separately and always assigned a fixed ID of `1` per frame.
             - Goalkeepers are tracked as a separate category (unlike ByteTrack where they are merged with players).
             - BoT-SORT uses Re-ID embeddings (via `osnet_x0_25_msmt17.pt`) to improve re-identification of players.
             """

        if read_from_cache and cache_path is not None and os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        tracking={
            "players":[],
            "referees":[],
            "ball":[],
            "goalkeeper":[]
        }
        
        detection_results = self.detect_frames(frames) 

        cls_names={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        cls_names_inv = {v:k for k,v in cls_names.items()}

#  Loop through each frame and corresponding detections
        for frame_idx,(frame, result) in enumerate(zip(frames, detection_results)):
            boxes = result.boxes  # Assume result has .boxes just like in single-frame prediction

            # Format detections: [x1, y1, x2, y2, conf, cls]
            dets = self.format_detection(boxes, confidence_threshold=confidence_threshold)
            # Step 3: Track
            tracks = self.tracker2.update(dets.copy(), frame)

            tracking["players"].append({})
            tracking["referees"].append({})
            tracking["ball"].append({})
            tracking["goalkeeper"].append({})

            for tracked_frame in tracks:
                bbox=tracked_frame[:4].tolist()
                cls_id = tracked_frame[6]
                track_id = tracked_frame[4]

                if cls_id == cls_names_inv['player']:
                    tracking["players"][frame_idx][track_id] = {"bbox":bbox}
                
                elif cls_id == cls_names_inv['referee']:
                    tracking["referees"][frame_idx][track_id] = {"bbox":bbox}
                
                elif cls_id ==cls_names_inv["goalkeeper"]:
                    tracking["goalkeeper"][frame_idx][track_id] = {"bbox":bbox}
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == cls_names_inv['ball']:
                    tracking["ball"][frame_idx][1] = {"bbox": box.xyxy[0].tolist()}  # Use fixed ID 1 for ball
                    break  # Use only one ball (if multiple detected)
        if cache_path is not None:
            with open(cache_path,'wb') as f:
                pickle.dump(tracking,f)
        return tracking    


    def object_tracker(self,tracker_name,frames, confidence_threshold,read_from_cache=False, cache_path=None):
        """
    Selects and runs the specified object tracking algorithm on a sequence of video frames.

    Args:
        tracker_name (str): 
            Name of the tracker to use. Must be one of ["byteTracker", "botSORT"].
        frames (list of np.ndarray): 
            List of video frames to perform tracking on.
        confidence_threshold (float): 
            Minimum confidence for detection (used in BoT-SORT only).
        read_from_cache (bool, optional): 
            If True and cache exists, loads tracking result from `cache_path`. Defaults to False.
        cache_path (str, optional): 
            Path to the pickle file to save/load cached tracking data. Defaults to None.

    Returns:
        dict: A dictionary containing per-frame tracking results structured as:
            {
                "players":    [ {track_id: {"bbox": [...]}} ],
                "referees":   [ {...} ],
                "ball":       [ {...} ],
                "goalkeeper": [ {...} ]
            }

    Raises:
        ValueError: If `tracker_name` is not one of the supported trackers.
        """
        if tracker_name == "byteTracker":
            return self.get_object_track_with_byteTracker(frames=frames, read_from_cache=read_from_cache, cache_path=cache_path)
        elif tracker_name=="botSORT":
            return self.get_object_track_with_botSORT(frames=frames, confidence_threshold=confidence_threshold,read_from_cache=read_from_cache, cache_path=cache_path)
        else:
            raise ValueError("Tracker not found")
        
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        """
        Draws an ellipse (used for players and referees) and optionally displays the track ID on the frame.

    Args:
        frame (np.ndarray): 
            Frame image on which to draw the ellipse.
        bbox (list or tuple): 
            Bounding box in the format [x1, y1, x2, y2].
        color (tuple): 
            BGR color tuple for the ellipse and ID box.
        track_id (int, optional): 
            If provided, draws a small filled rectangle with the ID number.

    Returns:
        np.ndarray: 
            Annotated frame with ellipse and optional ID label.

    Notes:
        - Ellipse is centered at the bottom of the bounding box.
        - The track ID box appears just below the ellipse.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
            
    def draw_traingle(self, frame, bbox, color):
        """
    Draws a filled triangle above the bounding box to visually represent the tracked ball.

    Args:
        frame (np.ndarray): 
            The frame image on which to draw the triangle.
        bbox (list or tuple): 
            Bounding box coordinates for the ball in the format [x1, y1, x2, y2].
        color (tuple): 
            BGR color tuple for the filled triangle (e.g., (0, 255, 0) for green).

    Returns:
        np.ndarray: 
            Annotated frame with the triangle drawn on top of the ball's bounding box.

    Notes:
        - The triangle points upward and is centered horizontally based on the bounding box.
        - Useful for distinguishing the ball from other tracked objects like players and referees.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks):
        """
   Draws visual annotations (ellipses for players/referees, triangle for ball) 
    on a list of video frames based on the provided tracking results.

    Args:
        video_frames (list of np.ndarray): 
            List of original video frames to annotate.
        tracks (dict): 
            A dictionary of tracking results per frame, with keys: "players", "referees", "ball".

    Returns:
        list of np.ndarray: 
            Annotated video frames with ellipses and triangles drawn on top of tracked objects.

    Notes:
        - Players are drawn as colored ellipses with tracking IDs.
        - Referees are drawn in yellow; ball is drawn as a green triangle.
        - Goalkeeper drawing is commented out by default.
        """
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # goalkeeper_dict = tracks["goalkeeper"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, int(track_id))

            # Draw Referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255),int(track_id))
            
            #draw goalkeeper
            # for track_id, goalie in goalkeeper_dict.items():
            #     frame = self.draw_ellipse(frame, goalie["bbox"],(115, 147, 179),int(track_id))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))



            output_video_frames.append(frame)

        return output_video_frames






