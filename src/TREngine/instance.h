#include <opencv2/imgproc.hpp>
cv::Mat get_person_affine_transform(float bbox[], bool inv = false);
void drawPostures(float *key_points, float *person_boxes, int box_count, cv::Mat &img);
void drawFacehead(float *facehead_boxes, int box_count, cv::Mat &img);
void get_out_bbox(float src[], int dst[]);