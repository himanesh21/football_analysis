import numpy as np 
import cv2

class ViewTransformer():
    """
    Transforms 2D pixel coordinates from a broadcast camera view to real-world court coordinates
    using a homographic transformation matrix. Useful for player and ball trajectory analysis
    in sports analytics.

    The transformation assumes the input frame comes from a fixed-angle camera where
    a known 4-point mapping (pixel to world) can be defined.

    Attributes
    ----------
    pixel_vertices : np.ndarray
        4 reference points in the image (in pixels) forming a quadrilateral on the court.

    target_vertices : np.ndarray
        Corresponding 4 points in the real-world top-down view (in meters).

    persepctive_trasnformer : np.ndarray
        3x3 homography matrix for transforming pixel coordinates to real-world coordinates.

    Example
    -------
    >>> vt = ViewTransformer()
    >>> point_on_frame = np.array([400, 500])
    >>> real_world_point = vt.transform_point(point_on_frame)
    >>> vt.add_transformed_position_to_tracks(tracks)
    """
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        """
    Transforms a single point from image space to real-world space using homography.

    Parameters
    ----------
    point : np.ndarray
        A NumPy array of shape (2,) representing the (x, y) pixel coordinate.

    Returns
    -------
    np.ndarray or None
        Transformed (x, y) real-world coordinate if the point is inside the source polygon.
        Returns None if the input point is outside the region defined by `pixel_vertices`.

    Notes
    -----
    - Points are first checked against the image polygon before transforming.
    - Useful for ensuring only valid court region points are mapped.
        """
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        """
    Adds real-world court coordinates to tracked object positions.

    For each tracked object's adjusted position (after camera motion compensation),
    this method transforms it using the perspective matrix and adds the result to
    the track under the key `'position_transformed'`.

    Parameters
    ----------
    tracks : dict
        A nested dictionary of the format:
        {
            "players": [
                {id1: {"position_adjusted": (x, y), ...}, ...},  # frame 0
                ...
            ],
            ...
        }

    Modifies
    --------
    - Adds `'position_transformed'` key for each valid track dictionary in-place.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if track is None:
                    continue
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed