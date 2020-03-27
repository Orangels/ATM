#include "instance.h"

// #ifndef INSTANCE_H
// #define INSTANCE_H
#include<iostream>
#include<stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <string>

using namespace std;

cv::Point2f get_3rd_point(cv::Point2f a, cv::Point2f b) {
	cv::Point2f direct = a - b;
	cv::Point2f res = cv::Point2f(b.x - direct.y, b.y + direct.x);
	return res;
}

cv::Mat get_affine_transform_from_center_scale(float *center, float *scale, float rot, bool inv) {
	float keypoint_pixel_std = 200.0;
	float output_size[2] = { 192, 256 };
	if (inv) {
		output_size[0] = 48;
		output_size[1] = 64;
	}
	float shift[2] = { 0, 0 };
	float scale_tmp[2] = { scale[0] * keypoint_pixel_std, scale[1] * keypoint_pixel_std };
	float src_w = scale_tmp[0];
	float dst_w = output_size[0];
	float dst_h = output_size[1];
	float rot_rad = 3.14159 * rot / 180;
	float src_dir_tmp[2] = { 0, -src_w * 0.5f };
	float src_dir[2] = { src_dir_tmp[0] * cos(rot_rad) - src_dir_tmp[1] * sin(rot_rad),
		src_dir_tmp[0] * sin(rot_rad) + src_dir_tmp[1] * cos(rot_rad) };
	float dst_dir[2] = { 0, -dst_w * 0.5f };
	cv::Point2f src[3];
	cv::Point2f dst[3];
	src[0] = cv::Point2f(center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]);
	src[1] = cv::Point2f(center[0] + src_dir[0] + scale_tmp[0] * shift[0],
		center[1] + src_dir[1] + scale_tmp[1] * shift[1]);
	dst[0] = cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);
	dst[1] = cv::Point2f(dst_w * 0.5f + dst_dir[0], dst_h * 0.5f + dst_dir[1]);
	src[2] = get_3rd_point(src[0], src[1]);
	dst[2] = get_3rd_point(dst[0], dst[1]);
	cv::Mat trans;
	if (inv)
		trans = cv::getAffineTransform(dst, src);
	else
		trans = cv::getAffineTransform(src, dst);
	return trans;
}

float *get_center_scale(float bbox[]) {
	const float aspect_ratio = (192 * 1.0 / 256); //��߱�
	float keypoint_pixel_std = 200.0; //cfg.KEYPOINT.PIXEL_STD
	float x = bbox[0];
	float y = bbox[1];
	float w = bbox[2] - bbox[0];
	float h = bbox[3] - bbox[1];
	float center[2] = { x + w * 0.5f, y + h * 0.5f };
	if (w > aspect_ratio * h)
		h = w * 1.0 / aspect_ratio;
	else
		w = h * aspect_ratio;
	float scale[2] = { w * 1.0f / keypoint_pixel_std, h * 1.0f / keypoint_pixel_std };
	if (center[0] != -1) {
		scale[0] = scale[0] * 1.25;
		scale[1] = scale[1] * 1.25;
	}
	static float center_scale[4];
	memcpy(center_scale, center, sizeof(center));
	memcpy(center_scale + 2, scale, sizeof(scale));
	return center_scale;
}

cv::Mat get_person_affine_transform(float bbox[], bool inv) {
	float *center_scale = get_center_scale(bbox);
	float center[2] = { center_scale[0], center_scale[1] };
	float scale[2] = { center_scale[2], center_scale[3] };
	float rotate = 0.0f;

	cv::Mat trans = get_affine_transform_from_center_scale(center, scale, rotate, inv);
	return trans;
}

//vkeypints: (n*17*64*48)*vboxCount
//vppersonBoxes: (classId,score, xmin,ymin,xmax,ymax)*vboxCount;(n*6)
//void drawPostures(cv::Mat& vioimg, float* vppersonBoxes, int vboxCount, float* vkeypints);
void
get_max_preds(float *key_points, int box_count, vector<vector<float>> &max_vals, vector<vector<vector<int>>> &preds) {
	const int num_joints = 17;
	const int width = 48;
	const int height = 64;
	const int area = width * height;//(width*height)
	const int single_size = width * height * num_joints;
	int temp[area];
	for (int id = 0; id < box_count; id++) {
		vector<float> max_val;
		vector<vector<int>> pred;
		for (int i = 0; i < 17; i++) {
			memcpy(temp, key_points + i * area + id * single_size, sizeof(temp));


			int index = max_element(temp, temp + area) - temp;
			//float max_value = *max_element(temp, temp + area); //why wrong
			float max_value = key_points[index + i * area + id * single_size];
			std::cout<<i<<": "<< (max_value)<< "   "<< key_points[index  + i * area + id * single_size]<<std::endl;
			max_val.push_back(max_value);
			vector<int> point;
			if (max_value > 0.4)
				point = { index % width, index / width };
			else
				point = { -1, -1 };
			pred.push_back(point);
			cout<<point[0]<<" "<<point[1]<<std::endl;
		}
		preds.push_back(pred);
		max_vals.push_back(max_val);
	}
}

void transform_preds(vector<vector<vector<int>>> &preds, vector<cv::Mat> &trans) {

	for (int img_id = 0; img_id < preds.size(); img_id++) {
		cv::Mat tran = trans[img_id];
		for (int point_id = 0; point_id < 17; point_id++) {
			int pred_x = preds[img_id][point_id][0];
			int pred_y = preds[img_id][point_id][1];
			if (pred_x == -1 | pred_y == -1)
				continue;
			int new_x = int(pred_x * tran.at<double>(0, 0) + pred_y * tran.at<double>(0, 1) + tran.at<double>(0, 2));
			int new_y = int(pred_x * tran.at<double>(1, 0) + pred_y * tran.at<double>(1, 1) + tran.at<double>(1, 2));
			preds[img_id][point_id][0] = new_x;
			preds[img_id][point_id][1] = new_y;
		}
	}
}

const int keypoint_flip_map[15][2] = { { 1,  2 },
{ 1,  0 },
{ 2,  0 },
{ 2,  4 },
{ 1,  3 },
{ 6,  8 },
{ 8,  10 },
{ 5,  7 },
{ 7,  9 },
{ 12, 14 },
{ 14, 16 },
{ 11, 13 },
{ 13, 15 },
{ 6,  5 },
{ 12, 11 } };
const vector<cv::Scalar> color_line = { cv::Scalar(255, 0, 127), cv::Scalar(253, 49, 95), cv::Scalar(250, 97, 63),
cv::Scalar(243, 142, 31), cv::Scalar(235, 180, 0), cv::Scalar(224, 212, 32),
cv::Scalar(211, 236, 64), cv::Scalar(196, 250, 96), cv::Scalar(179, 254, 128),
cv::Scalar(161, 249, 160), cv::Scalar(140, 234, 192), cv::Scalar(119, 210, 224),
cv::Scalar(96, 178, 255), cv::Scalar(72, 139, 255), cv::Scalar(48, 95, 255),
cv::Scalar(23, 46, 255), cv::Scalar(0, 0, 255) };

void drawPostures(float *key_points, float *person_boxes, int box_count, cv::Mat &img) {
	const float vis_th = 0.3;
	vector<cv::Mat> trans;
	float single_person_box[4];
	for (int i = 0; i < box_count; i++) {
		memcpy(single_person_box, person_boxes + i * 6 + 2, sizeof(single_person_box));
		trans.push_back(get_person_affine_transform(single_person_box, true));
		cv::imwrite("trans"+ std::to_string(i)+".jpg", trans[i]);
	}
	vector<vector<float>> max_vals;
	vector<vector<vector<int>>> preds;
	get_max_preds(key_points, box_count, max_vals, preds);
	transform_preds(preds, trans);
	for (int i = 0; i < box_count; i++) {
		float score = *(person_boxes + i * 6 + 1);
		if (score < vis_th)
			continue;
		int class_id = int(*(person_boxes + i * 6 + 0));
		int bbox_x = int(*(person_boxes + i * 6 + 2));
		int bbox_y = int(*(person_boxes + i * 6 + 3));
		int bbox_x2 = int(*(person_boxes + i * 6 + 4));
		int bbox_y2 = int(*(person_boxes + i * 6 + 5));
		//draw bbox
		cv::rectangle(img, cv::Point(bbox_x, bbox_y), cv::Point(bbox_x2, bbox_y2), cv::Scalar(0, 255, 255), 2, 8, 0);
		//draw key_point line
		vector<vector<int>> pred = preds[i];
		for (int j = 0; j < 15; j++) {
			int point_id1 = keypoint_flip_map[j][0];
			int point_id2 = keypoint_flip_map[j][1];
			if (pred[point_id1][0] != -1 & pred[point_id2][0] != -1)
				cv::line(img, cv::Point(pred[point_id1][0], pred[point_id1][1]),
					cv::Point(pred[point_id2][0], pred[point_id2][1]), color_line[j], 4, 16, 0);
		}
		// Draw mid shoulder / mid hip first for better visualization.
		//mid_shoulder: 6 5  sc_mid_hip: 12 11
		int left_shoulder_x = pred[5][0];
		int right_shoulder_x = pred[6][0];
		int nose_x = pred[0][0];
		cv::Point mid_shoulder;
		if (left_shoulder_x == -1 || right_shoulder_x == -1 || nose_x == -1)
			continue;
		int left_shoulder_y = pred[5][1];
		int right_shoulder_y = pred[6][1];
		mid_shoulder = cv::Point((left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2);
		cv::Point nose(pred[0][0], pred[0][1]);
		cv::line(img, mid_shoulder, nose, color_line[15], 4, 16, 0);
		int left_hip_x = pred[11][0];
		int right_hip_x = pred[12][0];
		if (left_hip_x == -1 || right_hip_x == -1)
			continue;
		int left_hip_y = pred[11][1];
		int right_hip_y = pred[12][1];
		cv::Point mid_hip((left_hip_x + right_hip_x) / 2, (left_hip_y + right_hip_y) / 2);
		cv::line(img, mid_hip, mid_shoulder, color_line[15], 4, 16, 0);
	}
}
void drawFacehead(float *facehead_boxes, int box_count, cv::Mat &img)
{
    vector<cv::Mat> trans;
	float single_person_box[4];
	for (int i = 0; i < box_count; i++) {
		memcpy(single_person_box, facehead_boxes + i * 6 + 2, sizeof(single_person_box));
//		trans.push_back(get_person_affine_transform(single_person_box, true));
	}
	for (int i = 0; i < box_count; i++) {
		float score = *(facehead_boxes + i * 6 + 1);
		int class_id = int(*(facehead_boxes + i * 6 + 0));
		int bbox_x = int(*(facehead_boxes + i * 6 + 2));
		int bbox_y = int(*(facehead_boxes + i * 6 + 3));
		int bbox_x2 = int(*(facehead_boxes + i * 6 + 4));
		int bbox_y2 = int(*(facehead_boxes + i * 6 + 5));
		cv::rectangle(img, cv::Point(bbox_x, bbox_y), cv::Point(bbox_x2, bbox_y2), cv::Scalar(0, 255, 255), 2, 8, 0);
    }
}
// bool readBinaryDataFromFile2Memory(const std::string &vFileName, void *&vopBuffer, size_t &vioBites) {
// 	std::ifstream Fin(vFileName, std::ios::binary);
// 	if (Fin.fail()) {
// 		std::cout << "Fail to read data from file " << vFileName << std::endl;;
// 		return false;
// 	}
// 	if (vioBites < 1) {
// 		Fin.seekg(0, std::ios_base::end);
// 		vioBites = Fin.tellg();
// 		Fin.seekg(0, std::ios_base::beg);
// 	}
// 	if (NULL == vopBuffer) vopBuffer = malloc(vioBites);
// 	Fin.read((char *)vopBuffer, vioBites);
// 	Fin.close();
// 	return true;
// }

// #endif // INSTANCE_H
void get_out_bbox(float src[], int dst[]) {
	const float aspect_ratio = (192 * 1.0 / 256); //��߱�
	float pad_scale = 1.25; //cfg.KEYPOINT.PIXEL_STD
	float x = src[0];
	float y = src[1];
	float w = src[2] - src[0];
	float h = src[3] - src[1];
	float center[2] = { x + w * 0.5f, y + h * 0.5f };
	if (w > aspect_ratio * h)
		h = w * 1.0 / aspect_ratio;
	else
		w = h * aspect_ratio;
	w = w * pad_scale;
	h = h * pad_scale;
	x = center[0] - w / 2;
	y = center[1] - h / 2;
	dst[0] = x;
	dst[1] = y;
	dst[2] = w;
	dst[3] = h;
}