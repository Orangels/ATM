//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_ENTRY_DETECT_H
#define ATM_ENTRY_DETECT_H

#include <iostream>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class EntryDetect {
public:
    EntryDetect();
    ~EntryDetect();

    void update(InstanceGroup instance_group);
    bool entry_flag;
    int entry_times;
    vector<Box> entry_face;
};

#endif //ATM_ENTRY_DETECT_H
