import numpy as np
import pymaxflow

class GraphCut:
    def segment(self, image, fg_points, bg_points):
        eps = 1e-5
        inf = 1e9

        def create_graph(vertices_count, edges_count):
            return pymaxflow.PyGraph(vertices_count, edges_count)
        def add_nodes(graph, count):
            graph.add_node(count)
        def add_edges(graph, src_vertices, dst_vertices, weights, reverse_weights):
            graph.add_edge_vectorized(src_vertices, dst_vertices,
                weights, reverse_weights)
        def set_foreground_weights(graph, points, indices_array, weight=1e10):
            indices = indices_array[points[:, 1], points[:, 0]].ravel()
            graph.add_tweights_vectorized(indices,
                np.zeros(len(points), np.float32),
                np.full(len(points), weight, dtype=np.float32))
        def set_background_weights(graph, points, indices_array, weight=1e10):
            indices = indices_array[points[:, 1], points[:, 0]].ravel()
            graph.add_tweights_vectorized(indices,
                np.full(len(points), weight, dtype=np.float32),
                np.zeros(len(points), np.float32))
        def set_term_weights(graph, indices, weights_up, weights_down):
            graph.add_tweights_vectorized(indices, weights_up, weights_down)
        def compute_adj_dist(v1, v2):
            rsigma2 = 1.0 / 10.0
            return np.exp(- 0.5 * rsigma2 * np.abs(v1 - v2))
        def compute_term_weights(img, fg_points, bg_points):
            fg_hist, fg_bins = np.histogram(img[fg_points[:, 1], fg_points[:, 0]],
                bins=8, range=(0, 256), density=True)
            bg_hist, bg_bins = np.histogram(img[bg_points[:, 1], bg_points[:, 0]],
                bins=8, range=(0, 256), density=True)

            fg_ind = np.searchsorted(fg_bins[1:-1], img.flatten())
            bg_ind = np.searchsorted(bg_bins[1:-1], img.flatten())

            fg_prob = fg_hist[fg_ind] + eps
            bg_prob = bg_hist[bg_ind] + eps

            scale = 1.0
            fg_weights = -scale * np.log(fg_prob)
            bg_weights = -scale * np.log(bg_prob)
            return fg_weights, bg_weights

        image = image.convert('L')
        im = np.array(image)

        indices = np.arange(im.size, dtype=np.int32).reshape(im.shape)
        fg_points = np.array(fg_points, dtype=int)
        bg_points = np.array(bg_points, dtype=int)

        g = create_graph(im.size, im.size * 4)
        add_nodes(g, im.size)

        # adjacent down
        diffs = (compute_adj_dist(im[:, 1:], im[:, :-1]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:,  :-1].ravel()
        e2 = indices[:, 1:  ].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent up
        diffs = (compute_adj_dist(im[:, :-1], im[:, 1:]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:, :-1].ravel()
        e2 = indices[:, 1:].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent right
        diffs = (compute_adj_dist(im[1:, 1:], im[:-1, :-1]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[1:,    :-1].ravel()
        e2 = indices[ :-1, 1:  ].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent left
        diffs = (compute_adj_dist(im[:-1, :-1], im[1:, 1:]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:-1, :-1].ravel()
        e2 = indices[1:, 1:].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        fg_weights, bg_weights = compute_term_weights(im, fg_points, bg_points)
        set_term_weights(g, indices.ravel(),
            fg_weights.astype(np.float32).ravel(),
            bg_weights.astype(np.float32).ravel())

        # links the to source and sink
        set_foreground_weights(g, fg_points, indices, inf)
        set_background_weights(g, bg_points, indices, inf)

        g.maxflow()

        out = g.what_segment_vectorized()
        return out.reshape(im.shape)