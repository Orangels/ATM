//
// Created by 王智慧 on 2020/3/28.
//

#include "structures/face_library.h"
#include "config.h"
#include <math.h>

FaceLibrary::FaceLibrary() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    duration = stoi(labels["FACELIB_DURATION"]) * 60;
    match_th = stof(labels["FACELIB_MATCH_TH"]);
    login_th = stof(labels["FACELIB_LOGIN_TH"]);
    ignore_th = (match_th + login_th) / 2;
    ind = 0;
}

FaceLibrary::~FaceLibrary() = default;

void FaceLibrary::time_filter(long time){
    vector<int> delete_ids;
    for(auto &ins : face_list){
        if (time - ins.second.time > duration){
            delete_ids.push_back(ins.first);
        }
    }
    for (auto &id : delete_ids){
        face_list.erase(id);
    }
}

void FaceLibrary::get_identity(vector<vector<float>> feas, vector<int> &face_ids){
    struct timeval now_time;
    gettimeofday(&now_time, NULL);
//    time_filter(now_time.tv_sec);

    int id, num_fea = feas.size();
    float dis, dis_max;

    if (num_fea > 0){
        if (face_list.size() > 0){
            for (auto fea : feas){
                for(auto ins : face_list){
                    dis = cosine(fea, ins.second.face_fea);
                    if (dis > dis_max){
                        id = ins.first;
                        dis_max = dis;
                    }
                }
                if (dis_max >= match_th){
                    face_list[id].time = now_time.tv_sec;
                    face_ids.push_back(id);
                } else if(dis_max < login_th){
                    InstenceFace insface;
                    insface.time = now_time.tv_sec;
                    insface.face_fea = fea;
                    face_list[ind] = insface;
                    face_ids.push_back(ind);
                    ind++;
                } else{
                    face_ids.push_back(-1);
                }
            }
        } else{
            for (auto fea : feas){
                InstenceFace insface;
                insface.time = now_time.tv_sec;
                insface.face_fea = fea;
                face_list[ind] = insface;
                face_ids.push_back(ind);
                ind++;
            }
        }
    }
}

float dotProduct(const vector<float>& v1, const vector<float>& v2)
{
    assert(v1.size() == v2.size());
    float ret = 0.0;
    for (vector<float>::size_type i = 0; i != v1.size(); ++i)
    {
        ret += v1[i] * v2[i];
    }
    return ret;
}
float module(const vector<float>& v)
{
    float ret = 0.0;
    for (vector<float>::size_type i = 0; i != v.size(); ++i)
    {
        ret += v[i] * v[i];
    }
    return sqrt(ret);
}
float cosine(const vector<float>& v1, const vector<float>& v2)
{
    assert(v1.size() == v2.size());
    return dotProduct(v1, v2) / (module(v1) * module(v2));
}