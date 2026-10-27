import numpy as np

def parse_chunks(frame, boxes, min_len=16):
    frame_chunks, boxes_chunks = [], []
    frame = np.asarray(frame); boxes = np.asarray(boxes)
    if len(frame) == 0: return frame_chunks, boxes_chunks
    breaks = np.where(np.concatenate([[0], frame[1:] - frame[:-1]]) != 1)[0]
    start = 0
    for bk_i, bk in enumerate(breaks):
        end = int(bk)
        if end > start and end-start >= min_len:
            frame_chunks.append(frame[start:end]); boxes_chunks.append(boxes[start:end])
        start = end
        if bk_i == len(breaks)-1:
            if len(frame)-start >= min_len:
                frame_chunks.append(frame[start:]); boxes_chunks.append(boxes[start:])
    return frame_chunks, boxes_chunks
