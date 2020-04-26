class Node:
    def __init__(self, box, frame_id, next_fram_id=-1):
        self.box = box
        self.frame_id = frame_id
        self.next_frame_id = next_fram_id


class Track:
    def __init__(self, id):
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks:
    def __init__(self):
        self.tracks = list()

    def add_node(self, node, id):
        node_added = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_added = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_added:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]
