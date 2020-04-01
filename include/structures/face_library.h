//
// Created by 王智慧 on 2020/3/28.
//

#ifndef ATM_FACE_LIBRARY_H
#define ATM_FACE_LIBRARY_H

#include <vector>
#include <unordered_map>
#include <sys/time.h>
#include <assert.h>
#include "structures/structs.h"

using namespace std;

class FaceLibrary {
public:
    FaceLibrary();
    ~FaceLibrary();

    int duration, ind;
    float match_th, login_th, ignore_th;

    void time_filter(long time);
    void get_identity(vector<vector<float>> feas, vector<int> &face_ids);

    unordered_map<int, InstenceFace> face_list;
};

float cosine(const vector<float>& v1, const vector<float>& v2);

#endif //ATM_FACE_LIBRARY_H