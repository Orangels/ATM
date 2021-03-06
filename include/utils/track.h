//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_TRACK_H
#define ATM_TRACK_H

#include <vector>
#include "box_tracking.h"
#include "structures/structs.h"

class Track {
public:
    Track(int head_track_mistimes, int w, int h);
    ~Track();

    void run(vector<Box>);

    int img_w, img_h;
    vector<Rect> detection_rects;
    vector<int> tracking_result, delete_tracking_id;
    BoxTracker* tracker;
};

#endif //ATM_TRACK_H
