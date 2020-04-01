//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_INSTANCE_H
#define ATM_INSTANCE_H

#include <vector>
#include <deque>
#include "structures/structs.h"

using namespace std;

class Instance {
public:
    Instance();
    Instance(int _track_id, Box box, Box face);
    ~Instance();

    void update(Box head, Box face);
    void update_hop(Box hat, Box glass, Box mask);

    int track_id, face_id;
    Box head_box;
    vector<Box> face_box, hat_box, glass_box, mask_box;
    Angle pos_angle;
};

#endif //ATM_INSTANCE_H
