//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_STRUCTS_H
#define ATM_STRUCTS_H

#include <vector>

using namespace std;

typedef struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
} Box;

typedef struct Angle {
    float Y;
    float P;
    float R;
} Angle;

typedef struct InstenceFace {
    long time;
    vector<float> face_fea;
} InstenceFace;

#endif //ATM_STRUCTS_H
