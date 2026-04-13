import numpy as np

class KDNode:
    def __init__(self, point, data=None, left=None, right=None, axis=0):
        self.point = point  # The feature vector [I, L, P, B]
        self.data = data    # Original window or metadata
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    def __init__(self, points, metadata=None):
        """
        points: List of feature vectors (numpy arrays)
        metadata: List of metadata corresponding to points
        """
        self.root = self._build(list(zip(points, metadata or [None]*len(points))), depth=0)

    def _build(self, points_with_meta, depth):
        if not points_with_meta:
            return None

        k = len(points_with_meta[0][0])
        axis = depth % k

        points_with_meta.sort(key=lambda x: x[0][axis])
        median = len(points_with_meta) // 2

        return KDNode(
            point=points_with_meta[median][0],
            data=points_with_meta[median][1],
            left=self._build(points_with_meta[:median], depth + 1),
            right=self._build(points_with_meta[median + 1:], depth + 1),
            axis=axis
        )

    def query(self, target, k=1):
        """
        Find k nearest neighbors to target.
        Returns list of (distance, node_data)
        """
        best_nodes = [] # List of (-dist, data) to use as max-heap

        def _search(node):
            if node is None:
                return

            dist = np.linalg.norm(node.point - target)
            
            # Maintain a max-heap of size k
            if len(best_nodes) < k:
                best_nodes.append((-dist, node.data))
                best_nodes.sort(key=lambda x: x[0], reverse=True)
            elif dist < -best_nodes[0][0]:
                best_nodes[0] = (-dist, node.data)
                best_nodes.sort(key=lambda x: x[0], reverse=True)

            axis = node.axis
            diff = target[axis] - node.point[axis]

            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left

            _search(near)

            if len(best_nodes) < k or abs(diff) < -best_nodes[0][0]:
                _search(far)

        _search(self.root)
        return sorted([( -d, data) for d, data in best_nodes])
