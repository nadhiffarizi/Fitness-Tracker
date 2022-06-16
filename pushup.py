import numpy as np


'''Code base for pushup repetition'''


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _convert_distancesxyz_magnitude(self, row):
        '''Convert the x, y, z distances into one magnitude value'''
        return np.sqrt(np.square(row[0]) + np.square(row[1]) + np.square(row[2]))

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

    def _get_angle_by_names(self, landmarks, name_first, name_mid, name_end):
        lmk_first = landmarks[self._landmark_names.index(name_first)]
        lmk_mid = landmarks[self._landmark_names.index(name_mid)]
        lmk_end = landmarks[self._landmark_names.index(name_end)]
        return self._calculate_angle(lmk_first, lmk_mid, lmk_end)

    def _calculate_angle(self, first, mid, end):
        a = np.array(first)  # First
        b = np.array(mid)  # Mid
        c = np.array(end)  # End
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 1) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Yang ditambahin fafa
            # Jarak 12-30 atau 11-29 (panjang badan dari pundak ke tumit)
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_heel'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_heel'),

            # Jarak 16-32 atau pergelangan ke jempol kaki
            self._get_distance_by_names(landmarks, 'left_wrist', 'left_foot_index'),
            self._get_distance_by_names(landmarks, 'right_wrist', 'right_foot_index')
        ])

        # convert to magnitude from xyz distances, shape output (N,)
        embedding = np.apply_along_axis(self._convert_distancesxyz_magnitude, axis=1, arr=embedding)

        # add angles
        angles = np.array([
            # Angle
            # Siku kanan dan kiri (shoulder-elbow-wrist)
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_elbow', 'right_wrist'),

            # #sudut badan atas (shoulder-hip-knee)
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_hip', 'left_knee'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_hip', 'right_knee'),

            # #sudut badan bawah (hip-knee-ankle)
            self._get_angle_by_names(landmarks, 'left_hip', 'left_knee', 'left_ankle'),
            self._get_angle_by_names(landmarks, 'right_hip', 'right_knee', 'right_ankle'),

            # #sudut ketek (elbow-shoulder-hip)
            self._get_angle_by_names(landmarks, 'left_elbow', 'left_shoulder', 'left_hip'),
            self._get_angle_by_names(landmarks, 'right_elbow', 'right_shoulder', 'right_hip'),

            # sudut pergelangan tangan (elbow-wrist-index_1)
            self._get_angle_by_names(landmarks, 'left_elbow', 'left_wrist', 'left_index_1'),
            self._get_angle_by_names(landmarks, 'right_elbow', 'right_wrist', 'right_index_1')
        ])

        # completes embedding
        embedding = np.append(embedding, angles, axis=0)

        return embedding


class KNNClassifier(object):
    '''This is a class for custom KNN. We implement KNN using weighted-KNN
      for number couple of features. The output will be dictionary_vote,
      confidence level, and top-K distances to each data points in the dataset'''

    def __init__(self, X, y, x_test, K):
        self.K = K
        self.x_test = x_test
        self.X = X
        self.y = y

    def __call__(self):
        ''' Call class as a function.
        returns dictionary_vote, confidence_level, and distances '''
        result_list = []

        for i, point in enumerate(self.X):
            distances = self.x_test - point
            abs_distance = np.sqrt(self.__weight_features__(distances))
            result_list.append((abs_distance, self.y[i]))

        # sorting the result
        sorted_result = sorted(result_list, key=lambda x: x[0])[:self.K]

        # classify
        classify_dict = self.__classify__(sorted_result)

        # Threshold for distances
        final_dict = self.__threshold__(sorted_result, classify_dict)
        return final_dict, sorted_result

    def __classify__(self, sorted_result):
        classify_dict = {"up": 0, "down": 0}

        for distance, class_code in sorted_result:
            if class_code == 1:
                classify_dict["up"] += 1
            elif class_code == 0:
                classify_dict["down"] += 1

        return classify_dict

    def __threshold__(self, sorted_result, classify_dict):
        '''the distance threshold for UP classification is 25 for each distances.
           The distance threshold for down classification is 50 for each distances.
           return: final_dict and confidence level'''

        truncated_dict = {'up': 0, 'down': 0}
        for distance, class_code in sorted_result:
            if class_code == 1 and distance > 50:
                # this condition we don't confident with the 'up' classification
                classify_dict["up"] -= 1
            elif class_code == 0 and distance > 100:
                # this condition we don't confident with the 'down' classification
                classify_dict["down"] -= 1

        # Calculate confidence level
        if classify_dict['up'] > classify_dict['down']:
            confidence_level = (classify_dict['up'] / self.K) * 100
        elif classify_dict['up'] < classify_dict['down']:
            confidence_level = (classify_dict['down'] / self.K) * 100
        else:
            confidence_level = 0  # if down and up has the same votes
        classify_dict['conf_level'] = confidence_level
        return classify_dict

    def __weight_features__(self, distances):
        '''this method used for weighting prominent features in the dataset'''

        '''array of features = ['center', 'lshoulder_lelbow',
           'rshoulder_relbow', 'lelbow_lwrist', 'relbow_rwrist', 'lhip_lknee',
           'rhip_rknee', 'lknee_lankle', 'rknee_rankle', 'lshoulder_lwrist',
           'rshoulder_rwrist', 'lhip_lankle', 'rhip_rankle', 'lhip_lwrist',
           'rhip_rwrist', 'lshoulder_lankle', 'rshoulder_rankle', 'lelbow_relbow',
           'lknee_rknee', 'lwrist_rwrist', 'lankle_rankle', 'lshouder_lheel',
           'rshoulder_rheel', 'lwrist_lfootindex', 'rwrist_rfootindex',
           'agl_lshoulder_lelbow_lwrist', 'agl_rshoulder_relbow_rwrist',
           'agl_lshoulder_lhip_lknee', 'agl_rshoulder_rhip_rknee',
           'agl_lhip_lknee_lankle', 'agl_rhip_rknee_rankle',
           'agl_lelbow_lshoulder_lhip', 'agl_relbow_rshoulder_rhip',
           'agl_lelbow_lwrist_lindex1', 'agl_relbow_rwrist_rindex1'] '''

        weight = np.array([1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1,
                           1, 1, 1, 20, 20, 5,
                           5, 3, 3, 5, 5, 1.2, 1.2])

        _sum = 0
        for i in range(len(distances)):
            _sum += np.square(distances[i] * 1 / weight[i])
        return _sum




